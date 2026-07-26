# GSplat-ROCm-RDNA4: 

## Native 3D Gaussian Splatting Training on AMD Radeon AI PRO R9700

### Build & Run your docker
```
docker build -f Dockerfile.gfx1201  -t gsplat-rocm:gfx1201-rocm72  .

sudo docker run --rm -it \
  --name gsplat-dev \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add video \
  --group-add render \
  --ipc=host \
  --network=host \
  --shm-size=16g \
  -v ~/datasets:/datasets \
  -v ~/gsplat-workspace:/home/$(whoami) \
  -w /opt/gsplat \
  -e HSA_OVERRIDE_GFX_VERSION=12.0.1 \
  gsplat-rocm:gfx1201-rocm72 \
  /bin/bash
```


### Check you platform
```
cd /opt/gsplat

python - <<'EOF'
import torch
import gsplat

print("torch:", torch.__version__)
print("HIP:", torch.version.hip)
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))
print("gsplat:", gsplat.__file__)
EOF

```
#### The expected output, this is matching our requirement:
```
torch: 2.9.1+rocm7.2.1.gitff65f5bc
HIP: 7.2.53211-e1a6bc5663
CUDA available: True
GPU: AMD Radeon AI PRO R9700
gsplat: /opt/gsplat/gsplat/__init__.py
```




### Training your 3D GS with below commands:
```
cd /opt/gsplat

HIP_VISIBLE_DEVICES=0 ROCR_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0 python examples/simple_trainer.py default   \
  --data_dir /datasets/bicycle  \
  --data_factor 8   \
  --result_dir ./results/r9700_test
```

### the expected result:
<img width="1236" height="411" alt="val_step6999_0000" src="https://github.com/user-attachments/assets/7a116c0b-22cd-441b-844c-7fe08ca56479" />
<img width="1236" height="411" alt="val_step29999_0015" src="https://github.com/user-attachments/assets/115650e9-0360-45e2-b286-58396b6e00e3" />
<img width="1236" height="411" alt="val_step29999_0023" src="https://github.com/user-attachments/assets/81ba956d-bda3-4229-b4b9-65392add8f2c" />



