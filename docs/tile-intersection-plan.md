# Tile intersection 的 Triton 优化：实施计划

针对 gsplat ROCm fork（`AMD-Ecosystem/gsplat @ b01acd4`）在 gfx1201 / wave32 上的
tile intersection 阶段。目标是把 `intersect_tile` + radix sort + `intersect_offset`
这条链路重写成 Triton + rocprim 的混合实现，并保持可逐位验证。

本文只描述设计与验收标准，不含实现。

---

## 1. 现状与基线

### 1.1 调用链

`gsplat.rendering.rasterization()` 里这一段（`rendering.py`，三处结构相同）：

```
tiles_per_gauss, isect_ids, flatten_ids = isect_tiles(means2d, radii, depths, ...)
isect_offsets = isect_offset_encode(isect_ids, I, tile_width, tile_height)
```

`isect_tiles` 转发到 C++ `intersect_tile`（`Intersect.cpp`），内部是四步：

1. `intersect_tile_kernel`（第一趟，`cum_tiles_per_gauss == nullptr`）：每线程一个高斯，
   由 `means2d`/`radii` 算 tile bbox，只写 `tiles_per_gauss[N]`。
2. `at::cumsum` + `.item()`：得到 `cum_tiles_per_gauss` 和 `n_isects`（一次 D2H 同步）。
3. `intersect_tile_kernel`（第二趟）：每线程串行遍历自己 bbox 内的全部 tile，写出
   `isect_ids[n_isects]`（int64）与 `flatten_ids[n_isects]`（int32）。
4. `radix_sort_double_buffer`：cub/rocprim `DeviceRadixSort::SortPairs`，
   `begin_bit=0`，`end_bit = 32 + tile_n_bits + image_n_bits`。

`intersect_offset_kernel` 再扫一遍排序后的 `isect_ids`，在 tile 边界处填 `offsets`。

### 1.2 key 编码

```
isect_id = (image_id << (32 + tile_n_bits)) | (tile_id << 32) | bitcast<uint32>(depth)
```

深度是 fp32 的**位模式零扩展**（`*(int32_t*)&depth` 再零扩展），不是数值排序。
对正深度两者一致；负深度会排到正深度之后且内部逆序。近平面裁剪保证了实际只有正深度，
但任何重写都必须沿用位模式语义，否则不是逐位等价。

### 1.3 radii 已经是 opacity-aware 紧包围盒

`ProjectionEWA3DGSFused.cu`：

```
extend   = min(3.33, sqrt(2 * ln(opacity / ALPHA_THRESHOLD)))
radius_x = ceil(extend * sqrt(covar2d[0][0]))
radius_y = ceil(extend * sqrt(covar2d[1][1]))
```

也就是说 "按不透明度收缩包围盒" 这个常见优化 **已经做过了**。剩下的空间只有
「轴对齐包围盒 → 椭圆本身」，见第 4 节。

### 1.4 测量基线

README 的 profile（500k 高斯 / 1920×1080 / SH3 / 50 iter / R9700）：

| | tile 8 | tile 16 |
|---|---|---|
| `intersect_tile_kernel`（两趟合计） | 527.4 ms → 10.55 ms/iter | 79.0 ms → 1.58 ms/iter |
| rocprim radix sort | 455.8 ms → 9.12 ms/iter | 140.8 ms → 2.82 ms/iter |
| 整步总时间 | 43.1 ms/iter | 28.7 ms/iter |

**tile 16 下 binning 占 15.3%，这是本计划在不触及光栅化时的收益天花板（1.18×）。**
唯一能突破这个上限的是第 4 节，因为它同时缩小 23% 占比的光栅化反向。

由排序耗时反推，tile 16 下 `n_isects` 量级约 8–12 M（下文估算统一按 10 M）。

### 1.5 慢在哪

第二趟的写入是「每线程一段连续」：

```
int64_t cur_idx = (idx == 0) ? 0 : cum_tiles_per_gauss[idx - 1];
for (i = tile_min.y; i < tile_max.y; ++i)
  for (j = tile_min.x; j < tile_max.x; ++j) {
    isect_ids[cur_idx] = ...; flatten_ids[cur_idx] = idx; ++cur_idx;
  }
```

wave32 内 32 条 lane 的写地址相隔 `tiles_per_gauss × 8B`，一条 store 最多触碰 32 条
cache line 而不是 4 条；同时内层循环次数在 wave 内从 1 到几百不等。

- 理想流量：10 M × 12 B ≈ 120 MB，在 644 GB/s 上约 **0.19 ms**；实测 **1.58 ms**，差 8×。
- 佐证：tile 8→16 时 pair 数只降约 3.2×（看排序），这个内核却降了 6.7×，超线性的部分
  正是发散与写放大。

排序侧的问题是 key 里带了 32 位深度，要排 ~46 位 ≈ 6 个 digit pass × 12 B/元素。

---

## 2. 总体架构

三步共用一条流水线骨架，后一步只替换其中一个环节：

```
                    B（第 3 节）                    C（第 4 节）
step 1  counts      = bbox 宽×高（逐元素闭式）      rows = bbox 行数（逐元素闭式）
step 2  —                                           run kernel: 每行解闭式 → (start, len)
step 3  cumsum + repeat_interleave 得到 owner 映射
step 4  emit kernel: 每个 lane 写一个 pair，全合并写
step 5  排序                                        D（第 5 节）替换
step 6  offsets                                     D 里用 searchsorted 替代
```

打包成独立包 `triisect/`，与 `triraster/` 同构：不改 gsplat 磁盘文件，
`install()` / `uninstall()` 在进程内切换，不满足快路径条件时透明回退 HIP。

### 2.1 接入点（与 triraster 不同，注意）

`rendering.py` 是 **按名字导入** 的：

```
from .cuda._wrapper import (..., isect_offset_encode, isect_tiles, ...)
```

而 `_RasterizeToPixels` 是在 `_wrapper` 内部按模块全局查找的。所以：

- triraster 重绑 `gsplat.cuda._wrapper._RasterizeToPixels` 有效；
- triisect **必须重绑 `gsplat.rendering.isect_tiles` 和 `gsplat.rendering.isect_offset_encode`**，
  只改 `_wrapper` 里的同名符号不会生效。
- 为了让直接调用 `_wrapper.isect_tiles` 的第三方代码也走新路径，两处都绑。

### 2.2 快路径条件（不满足即回退 HIP）

- `packed == False`（`means2d.dim() == 3`）
- `segmented == False`
- `I == 1`（多相机/batch 先不支持；key 里的 image 位与 segmented 排序另算）
- `means2d.dtype == float32`，`radii.dtype == int32`，CUDA/HIP 张量
- `n_isects > 0`

回退时直接调用原始 `isect_tiles`，保证行为完全不变。

---

## 3. 步骤 B：输出并行的 emit 内核（纯 Triton，逐位等价）

### 3.1 思路

把并行维度从「高斯」换成「输出槽位」。每条 lane 负责恰好一个 `(gaussian, tile)` 对，
写地址在 wave 内完全连续，负载天然均衡。发散和写放大同时消失。

### 3.2 索引推导

对高斯 `g`，沿用现有 bbox 计算（`floor`/`ceil` + clamp，必须逐位照抄，包括
`min(max(0, ...), tile_width)` 的顺序）：

```
bw = jmax - jmin,  bh = imax - imin,  count = bw * bh
```

原实现按「先行后列」顺序 emit，因此第 `k` 个输出（`k = out_idx - cum[g-1]`）对应

```
i = imin + k // bw
j = jmin + k %  bw
```

与原顺序逐位一致 → 输出的 `isect_ids` / `flatten_ids` 可以用 `torch.equal` 对拍。

### 3.3 owner 映射：三个候选

emit kernel 需要由 `out_idx` 反查 `g`。三种做法：

| | 机制 | 额外开销 | 复杂度 |
|---|---|---|---|
| M1 | 核内对 `cum_counts` 做二分 | ~21 次 gather/元素，约 1.7 GB L2 流量 | 低 |
| M2 | 按 bbox 大小分桶 + 定长展开，无查找 | 需要在 N 上做一次 partition | 高 |
| M3 | `torch.repeat_interleave(arange(N, int32), counts, output_size=n_isects)` | +40 MB 写 +40 MB 读 | 最低 |

**v1 选 M3。** 理由：ATen 的 `repeat_interleave` 本身是一个流式合并写内核，
把 10 M×4 B 的 owner 数组物化只要 ~0.12 ms，而它换来的是一个完全没有查找、没有分支的
emit kernel。`output_size=n_isects` 必须传，否则 ATen 会再做一次 D2H 同步
（`n_isects` 我们在 cumsum 后已经 `.item()` 过一次）。

代价核算：理想 120 MB → 实际约 200 MB → 约 0.32 ms，仍是当前 1.58 ms 的 ~5×。
若 profile 显示 owner 物化占比明显，再按 M1 融合进 kernel。

### 3.4 kernel 形态

```
grid = cdiv(n_isects, BLOCK)
每个 program:
  k      = pid*BLOCK + arange(BLOCK)                 mask = k < n_isects
  g      = load(owner + k)                            # int32
  base   = load(cum + g - 1)  (g==0 → 0)
  kl     = k - base
  bw, jmin, imin = 由 load(means2d+2g), load(radii+2g) 重算（不额外存）
  i, j   = imin + kl // bw, jmin + kl % bw
  tile   = i * tile_width + j
  depth  = bitcast<int32>(load(depths + g))           # 零扩展成 int64
  store(isect_ids + k, (tile << 32) | depth)
  store(flatten_ids + k, g)
```

要点：

- `kl // bw` 是逐元素变量除法。Triton 会生成整数除法；若成为瓶颈，改用
  `float32` 倒数 + 一次修正（`bw` 上限是 `tile_width ≤ 120`，`kl` 上限 ~2^20，
  fp32 精度足够但必须带修正分支）。这项做成 `constexpr` 开关，A/B 一次即可。
- `means2d`/`radii`/`depths` 的读取是按 `g` 的 gather，但同一 wave 内 `g` 高度重复
  （平均每高斯 ~20 个 pair），命中 L1；不必预先展开成 per-pair 数组。
- 零半径高斯（`radii <= 0`）在 step 1 就得到 `count = 0`，不会出现在 owner 里。
- autotune：`BLOCK ∈ {256, 512, 1024}`、`num_warps ∈ {1, 2, 4}`。这个 kernel 是纯
  带宽型，不像 triraster 那样受寄存器压力支配，配置空间可以小。

### 3.5 step 1（counts）

一行 torch 表达式即可（`floor`/`ceil` 用整数运算复刻，避免 fp 边界差异），
或者一个 trivial 的 Triton kernel。N=500k，两者都在 20 µs 量级。
优先用 Triton，理由是第 4 节要在同一个 kernel 里换成行数计算。

### 3.6 验收

`tests/isect_correctness_test.py`：对每个用例同时跑 HIP 与 Triton，
断言 `torch.equal` 于 `tiles_per_gauss`、`isect_ids`、`flatten_ids`、`isect_offsets`。

用例矩阵：

- `tile_size ∈ {8, 16}`
- 图像宽高不是 tile 的整数倍（如 618×411，`bicycle --data_factor 8` 的真实尺寸）
- 含大量零半径高斯（被 near_plane / radius_clip 裁掉的）
- 高斯完全在画面外、bbox 被 clamp 成空
- `n_isects == 0`（全部裁光）
- 单个高斯覆盖整幅图（bbox = 全部 tile）
- 触发回退的输入（packed / I>1 / segmented / fp64）→ 断言走的是 HIP 且结果不变

### 3.7 预期

tile 16：1.58 → ~0.35 ms/iter（−1.2 ms，整步 −4%）。
tile 8：10.55 → ~1.0 ms/iter（−9.5 ms）。

---

## 4. 步骤 C：精确椭圆-tile 判交

### 4.1 为什么值得做

蒙特卡洛估算（脚本见 4.6），在 gsplat **已有的** opacity-aware 紧 AABB 基础上，
再做精确椭圆判交还能去掉的 pair：

| 各向异性上限 | tile 8 保留 | tile 16 保留 |
|---|---|---|
| 2× | 76.3% | 80.0% |
| 8× | 66.0% | 72.5% |
| 20× | 59.2% | 67.6% |

即 **pair 数减少 20–40%**。它同时作用在 emit、排序、offset encode 和光栅化前/反向
四个地方，是唯一能突破 1.4 节那个 15% 上限的方案。

### 4.2 为什么它是输出位精确的

前向核对不达阈值的高斯是 `continue`，既不更新 `T`，也不更新 `cur_idx`：

```
float alpha = min(0.999f, opac * __expf(-sigma));
if (sigma < 0.f || alpha < ALPHA_THRESHOLD) { continue; }
```

所以从 tile 列表里删掉「该 tile 内所有像素中心都不达阈值」的高斯，
`render_colors` / `render_alphas` / `last_ids` **一个 bit 都不会变**；
反向同理，只有原子累加顺序变化（与 triraster 已接受的噪声同类）。

保守性必须靠两点保证：

1. 判据用像素中心矩形 `[x0+0.5, x0+T-0.5] × [y0+0.5, y0+T-0.5]`，
   而不是 tile 的几何矩形——前者才是前向真正采样的位置集合，且它是后者的子集，
   所以更紧，同时仍然是「∃ 像素中心达阈值」的**超集**（连续矩形上的最小值
   ≤ 离散像素中心上的最小值）。
2. 阈值放宽 `τ' = τ + ε`。前向用的是 `__expf`（快速近似指数），边界处与精确
   `exp` 有几个 ulp 的差；`ε ≈ 1e-3`（相对 τ 的量级）足够覆盖，代价是几乎不损失裁剪率。
   这一条是「provably safe」的关键，不能省。

### 4.3 数学

记 conic `(A, B, C) = (Σ⁻¹₀₀, Σ⁻¹₀₁, Σ⁻¹₁₁)`，`u = mean.x - px`，`v = mean.y - py`，

```
sigma(u, v) = 0.5*(A u² + C v²) + B u v          τ = ln(opacity / ALPHA_THRESHOLD)
```

对固定的一条 tile 行，`v` 落在条带 `[v_lo, v_hi]`。集合 `{sigma ≤ τ}` 是凸的（Σ⁻¹ 正定），
凸集与水平条带的交在 `u` 上的投影是**一个区间** → 每行是一段**连续的 tile run**。

区间端点闭式：

- 无约束分支：`∂sigma/∂v = C v + B u = 0 → v* = -B u / C`，代回得
  `sigma = 0.5 u² (A - B²/C) = 0.5 u² / Σ₀₀`，故全局 `|u| ≤ sqrt(2τ Σ₀₀)`。
  这正好等于 `radius_x`，可用作实现的自检。
- 端点分支：在 `v = v_e`（`v_lo` 或 `v_hi`）上解 `0.5 A u² + B v_e u + 0.5 C v_e² - τ = 0`：
  `u = (-B v_e ± sqrt(B² v_e² - A C v_e² + 2 A τ)) / A`，判别式 < 0 表示该边界线不与椭圆相交。
- 行区间 `[u_min, u_max]` = 取两个端点分支的解，再在 `v*(±u_global)` 落在条带内时
  并入全局极值点。共 2 个 sqrt + ~20 flop。

列范围：`px ∈ [mean.x - u_max, mean.x - u_min]` → `j` 从 `floor(...)` 到 `ceil(...)`，
再与 bbox 的 `[jmin, jmax)` 取交（保证不会因 fp 误差跑到 AABB 之外）。
判别式全负 ⇒ 该行 run 长度为 0，这就是省下来的部分（主要是四角）。

### 4.4 流水线变化

第 3 节的两级（gaussian → pair）变成三级（gaussian → row → pair）：

```
step 1  rows_per_gauss = imax - imin                     闭式，无循环
        cumsum → row_offsets, R                          R ≈ (1~3)·N
step 2  row_owner = repeat_interleave(arange(N), rows_per_gauss, output_size=R)
step 3  [Triton] run kernel: 每行一条 lane，解 4.3 的闭式，写 run_start[R], run_len[R]
        全部合并读写，零发散
step 4  cumsum(run_len) → pair_offsets, n_isects
        pair_owner = repeat_interleave(arange(R), run_len, output_size=n_isects)
step 5  [Triton] emit kernel: tile = run_start[r] + (k - pair_offsets[r])，g = row_owner[r]
step 6  tiles_per_gauss（API 兼容）= pair_offsets[row_offsets[g+1]] - pair_offsets[row_offsets[g]]
```

R 级数组约 1.5 M × 8 B ≈ 12 MB，两次 cumsum 都在小数组上，可忽略。
emit kernel 与第 3 节几乎相同，只是 `tile` 的来源从 `//`/`%` 换成 run 描述符，
反而更便宜（少了变量除法）。

**注意**：`tiles_per_gauss` 语义会变（现在返回的是精确判交后的数目）。
它被 `rasterization()` 返回给调用方，`strategy/` 里的致密化不依赖它，
但要在 release note 里写明，并提供 `triisect.install(exact=False)` 关掉 C 只留 B。

### 4.5 验收

- `n_isects` 严格变小，记录比例（对真实场景应落在 60–80% 区间，与 4.1 的估算对照）。
- **渲染输出 `torch.equal`**：`render_colors`、`render_alphas`、`last_ids` 与
  baseline 逐位相同。这是本步的核心门槛，比对 `isect_ids` 更有意义。
- 复跑 `tests/rasbwd_correctness_test.py --grad-bias`：per-Gaussian 梯度范数的
  相对误差与符号偏置应与现有水平同量级（~1e-7 中位数，偏置 ±3e-9）。
- 单独构造对抗用例：细长（20:1）且旋转 45° 的高斯、低不透明度（0.02）高斯、
  恰好切过 tile 角的高斯——逐 tile 与暴力像素中心枚举对拍。

### 4.6 估算脚本

`/tmp/isect_sim.py`（纯 Python 蒙特卡洛，凸二次型在矩形上的精确最小值 =
原点在盒内则为 0，否则取四条边上的一维最小值）。若要复现或换分布，建议移到
`tests/isect_cull_estimate.py` 并接受分布参数。

### 4.7 预期

tile 16，pair 数 −27% 时：emit −0.1 ms，排序 −0.2 ms（已做 D 的话），
光栅化前向 −0.15 ms、反向 −0.8~1.3 ms（反向按 pair 数近似线性）。
合计 **−1.1 ~ −1.6 ms/iter**。

---

## 5. 步骤 D：排序（不要用 Triton）

### 5.1 为什么不用 Triton

设备级 radix sort 需要 per-block 直方图 → 全局 scan → 带 rank 的 scatter，
Triton 既没有 grid sync，也没有廉价的「block 内按上千个 bin 求稳定 rank」的手段
（8 位 digit 就要 `[BLOCK, 256]` 的 one-hot tile）。rocprim 的实现已经接近带宽。
分段 bitonic 同理：段长可变且可达上万，写不过 `DeviceSegmentedRadixSort`。

### 5.2 真正的浪费：排序位宽

现在 key 是 64 位且低 32 位是深度，要排约 46 位 ≈ 6 个 digit pass × 12 B/元素。

**如果先把高斯按深度位模式排一次序（N=50 万，比 10 M 个 pair 便宜一个量级），
并按该顺序 emit，那么只排 13–14 位的 tile 位就够了。** cub / rocprim 的
`SortPairs` 是稳定排序，因此 tile 内的相对顺序 = emit 顺序 = 深度顺序，
最终排列与现在**完全一致**（含深度相等时按原索引的 tie-break，前提是深度预排序本身稳定）。

- 流量：2 pass × 8 B（int32 tile key + int32 id）对比 6 pass × 12 B，约 4×。
- 必须用**深度的 uint32 位模式**做预排序 key，不能用 `torch.sort(depths)`，
  否则负深度的次序与 1.2 节的语义不一致。
- 预排序产生置换 `perm`；`counts` / `cum` 都在置换后的顺序上算，
  `flatten_ids` 写的是 `perm[g]`（原始索引），`tiles_per_gauss` 需要 scatter 回原序。

### 5.3 两条落地路径

| | 做法 | 预期 | 代价 |
|---|---|---|---|
| D1（先做） | `torch.argsort(tile_ids_int32, stable=True)` + gather | key 4 B / value 8 B，约 3× 流量削减 | 纯 Python，半小时；gather 是半局部随机访问，需实测 |
| D2（目标） | 给 `IntersectTile.cu` 加 patch，暴露 `begin_bit=32` 与 int32 key 的排序入口 | 约 4× | ~20 行 patch，与仓库现有 `patches/` 风格一致 |

先做 D1 量一个数；只有当它明显不及理论值时才上 D2。

### 5.4 顺带的收益

emit 出的是独立的 `tile_ids`（int32）而不是 `isect_ids`（int64）：

- emit 写入从 12 B/pair 降到 8 B/pair；
- `isect_offsets` 直接由 `torch.searchsorted(tile_ids_sorted, arange(n_tiles))` 得到，
  `intersect_offset_kernel` 整个省掉，也不再需要把 int64 的 `isect_ids` 读一遍；
- `isect_ids` 张量可以完全不物化（10 M × 8 B = 80 MB 的分配与写入）。

但 `isect_tiles` 的公开签名要求返回 `isect_ids`。做法：返回一个惰性对象或者
在 `install()` 时同时替换 `isect_offset_encode`，让它接受 `tile_ids`；
调用方只有 `rendering.py` 的三处，且它只把 `isect_ids` 传给 `isect_offset_encode`。
两个符号一起替换即可保持一致，但要在文档里写明「安装后 `isect_ids` 的编码已改变」，
并让 `isect_offset_encode` 对 int64 输入自动走原路径。

### 5.5 验收

`isect_offsets` 与 `flatten_ids` 对 HIP baseline `torch.equal`。
这两个是光栅化唯一消费的东西，`isect_ids` 本身的编码变化不需要对拍。

### 5.6 预期

tile 16：2.82 → ~0.8 ms/iter，另加 offset encode 省掉的 ~0.15 ms，
减去深度预排序的 ~0.05 ms → **−2.1 ms/iter**。

---

## 6. 里程碑与总收益

以 tile 16 / 28.7 ms per step 为基准：

| 步骤 | 内容 | 预期 | 累计 |
|---|---|---|---|
| B | 输出并行 emit（纯 Triton，逐位等价） | −1.2 ms | 27.5 ms |
| D1 | `argsort(stable)` 探针，验证排序假设 | −1.0 ~ −2.1 ms | ~25.9 ms |
| D2 | rocprim `begin_bit` patch + int32 key + searchsorted offsets | 补齐到 −2.2 ms | 25.4 ms |
| C | 精确椭圆判交 | −1.1 ~ −1.6 ms | **~24 ms（1.20×）** |

另有一个副作用值得在 C 之后单独跑一次：tile 8 的 Triton 反向是 5.14 ms/iter，
比 tile 16 的 6.57 ms 更快，之前输就输在 binning（19.7 ms vs 4.4 ms）。
B+D 做完后两者会拉到同一量级，`--tile-size` 需要重新扫。

---

## 7. 交付物

| 路径 | 内容 |
|---|---|
| `triisect/` | 包本体，与 `triraster/` 同构（`_core.py` 内核、`_patch.py` monkeypatch、`README.md` 设计说明） |
| `tests/isect_correctness_test.py` | 3.6 / 4.5 / 5.5 的门槛；`--verify-configs` 覆盖全部 autotune 候选 |
| `tests/profile_trainer.py` | 新增 `--isect baseline\|triton`，默认 `baseline`；header 行回显实际解析到的实现，导入失败时 **abort 而不是静默回退** |
| `tests/run_simple_trainer.py` | 新增 `GSPLAT_ISECT` 环境变量 |
| `Dockerfile.gfx1201` | 与 triraster 并列安装 |
| `patches/` | D2 的 rocprim patch（若采用） |
| `README.md` | 新增 "Choosing the tile-intersection implementation" 小节 + 更新 "What we measured" |

---

## 8. 风险登记

| 风险 | 影响 | 缓解 |
|---|---|---|
| `repeat_interleave` 的 owner 数组吃掉一半收益 | B 只拿到 3× 而不是 5× | 先测；必要时切 M1（核内二分）或 M2（分桶展开） |
| 变量整数除法拖慢 emit | B 收益打折 | fp32 倒数 + 修正，做成 `constexpr` 开关 A/B |
| C 的闭式解在 fp32 下不够保守，漏掉一个 pair | 渲染输出改变，逐位对拍失败 | `τ + ε` 放宽；与 bbox 取交作为硬上界；对抗用例覆盖 |
| D 的稳定性假设不成立（某版 rocprim 不稳定） | 排序结果与 baseline 不一致 | 5.5 的 `torch.equal` 门槛会直接抓到；抓到就退回 64 位 key |
| 多相机 / packed / segmented 路径 | 静默走错分支 | 2.2 的快路径条件 + 测试里显式断言回退 |
| `tiles_per_gauss` 语义在 C 之后改变 | 下游若依赖它会行为变化 | 提供 `exact=False` 开关；release note 写明 |
