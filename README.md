# CG Lab Homework 6: Differentiable Mesh Fitting

本仓库实现实验六：使用 PyTorch3D 的可微软光栅化，将一个初始化球体网格通过多视角剪影监督优化成目标奶牛网格形状。

## 实验目标

- 理解软光栅化如何在 Mesh 边界附近提供非零梯度，缓解硬光栅化造成的梯度消失。
- 使用多视角二维剪影反推三维网格顶点偏移量。
- 通过 Laplacian smoothing、edge length penalty 和 normal consistency 三类正则项稳定网格优化，避免尖刺、交叉和局部最优。

## 环境配置

推荐使用 Conda 创建独立环境。`requirements.txt` 只包含可直接通过 PyPI 安装的通用依赖；PyTorch3D 的 wheel 与 Python、PyTorch、CUDA 版本强相关，不能简单写成 `pytorch3d` 放进 `requirements.txt`，否则很多平台会出现 “No matching distribution found for pytorch3d”。

```bash
conda create -n cg-lab6 python=3.10 -y
conda activate cg-lab6
pip install -r requirements.txt
```

随后请按你的平台单独安装 PyTorch3D。Conda 用户通常可使用类似命令，并需要把 `pytorch`、`pytorch-cuda`、`pytorch3d` 与 CUDA 版本替换成彼此兼容的组合：

```bash
conda install -c pytorch -c nvidia pytorch torchvision pytorch-cuda=11.8
conda install -c pytorch3d pytorch3d
```

如果使用 pip 或 Apple Silicon / Windows / 源码编译，请以 PyTorch3D 官方安装说明为准。脚本会在真正运行实验时检查 `torch` 与 `pytorch3d` 是否可导入；`--help` 不需要提前安装这些重依赖。

## 数据准备

脚本支持两种方式获得目标奶牛模型：

1. 显式指定本地 OBJ：

```bash
python fit_cow_silhouette.py --target-obj /path/to/cow.obj
```

2. 不传 `--target-obj` 时，脚本会尝试从 PyTorch3D 官方示例仓库下载 cow mesh 到 `data/cow_mesh/`。

## 运行实验

快速调试：

```bash
python fit_cow_silhouette.py --iters 50 --image-size 128 --num-views 8 --device cpu
```

GPU 完整优化示例：

```bash
python fit_cow_silhouette.py \
  --iters 1000 \
  --image-size 256 \
  --num-views 20 \
  --sphere-level 4 \
  --lr 0.01 \
  --device auto
```

输出文件默认写入 `outputs/`：

- `target_silhouettes.png`：目标奶牛的多视角剪影。
- `progress_*.png`：优化过程中的剪影对比图。
- `loss_curve.png`：总 loss 以及各子 loss 曲线。
- `deformed_mesh.obj`：最终优化得到的网格。

## 实现要点

- `SoftSilhouetteShader` 使用 `BlendParams(sigma, gamma)` 构造平滑边界概率，保证顶点移动时剪影 loss 可导。
- 可学习参数为 `deform_verts`，优化时通过 `source_mesh.offset_verts(deform_verts)` 得到当前网格。
- 总损失由剪影 MSE 与三类正则项构成：

```text
L_total = L_silhouette
        + w_lap * L_laplacian
        + w_edge * L_edge
        + w_normal * L_normal
```

## 可调参数建议

- 如果结果像“刺猬”，增大 `--w-lap`、`--w-edge` 或 `--w-normal`。
- 如果轮廓始终贴不上目标，增大迭代次数、学习率或视角数。
- 如果优化早期几乎不动，可适当增大 `--sigma`，让软边界更宽。
- 如果最终边界太模糊，可在后期降低 `--sigma` 后继续优化。
