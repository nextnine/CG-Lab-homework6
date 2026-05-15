#!/usr/bin/env python3
"""Fit an ico-sphere mesh to a target cow mesh from multi-view silhouettes.

The script follows the standard PyTorch3D differentiable-rendering pipeline:

1. Load or download the target cow mesh.
2. Render target silhouettes from uniformly spaced cameras.
3. Initialize an ico-sphere source mesh.
4. Optimize per-vertex offsets with silhouette MSE plus mesh regularizers.
5. Save progress images, a loss curve, and the final deformed OBJ.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import urllib.request
from pathlib import Path
from typing import Iterable


def require_package(module_name: str, install_hint: str) -> None:
    """Fail early with a clear message before importing optional dependencies."""

    if importlib.util.find_spec(module_name) is None:
        raise SystemExit(f"Missing dependency '{module_name}'. {install_hint}")


def load_runtime_dependencies() -> None:
    """Import heavy rendering dependencies after CLI parsing.

    Delaying these imports keeps `python fit_cow_silhouette.py --help` usable in
    a fresh checkout and avoids making `pip install -r requirements.txt` fail on
    platforms where PyTorch3D must be installed from a platform-specific wheel or
    built from source.
    """

    require_package("torch", "Install PyTorch, e.g. `pip install torch torchvision`.")
    require_package("pytorch3d", "Install PyTorch3D with a command matching your PyTorch/CUDA version.")
    require_package("matplotlib", "Install matplotlib with `pip install matplotlib`.")
    require_package("tqdm", "Install tqdm with `pip install tqdm`.")

    global plt, torch, F
    global load_objs_as_meshes, save_obj
    global mesh_edge_loss, mesh_laplacian_smoothing, mesh_normal_consistency
    global BlendParams, FoVPerspectiveCameras, MeshRasterizer, MeshRenderer
    global RasterizationSettings, SoftSilhouetteShader, look_at_view_transform
    global Meshes, ico_sphere, tqdm

    import matplotlib.pyplot as plt_import
    import torch as torch_import
    import torch.nn.functional as F_import
    from pytorch3d.io import load_objs_as_meshes as load_objs_as_meshes_import
    from pytorch3d.io import save_obj as save_obj_import
    from pytorch3d.loss import mesh_edge_loss as mesh_edge_loss_import
    from pytorch3d.loss import mesh_laplacian_smoothing as mesh_laplacian_smoothing_import
    from pytorch3d.loss import mesh_normal_consistency as mesh_normal_consistency_import
    from pytorch3d.renderer import BlendParams as BlendParams_import
    from pytorch3d.renderer import FoVPerspectiveCameras as FoVPerspectiveCameras_import
    from pytorch3d.renderer import MeshRasterizer as MeshRasterizer_import
    from pytorch3d.renderer import MeshRenderer as MeshRenderer_import
    from pytorch3d.renderer import RasterizationSettings as RasterizationSettings_import
    from pytorch3d.renderer import SoftSilhouetteShader as SoftSilhouetteShader_import
    from pytorch3d.renderer import look_at_view_transform as look_at_view_transform_import
    from pytorch3d.structures import Meshes as Meshes_import
    from pytorch3d.utils import ico_sphere as ico_sphere_import
    from tqdm import tqdm as tqdm_import

    plt = plt_import
    torch = torch_import
    F = F_import
    load_objs_as_meshes = load_objs_as_meshes_import
    save_obj = save_obj_import
    mesh_edge_loss = mesh_edge_loss_import
    mesh_laplacian_smoothing = mesh_laplacian_smoothing_import
    mesh_normal_consistency = mesh_normal_consistency_import
    BlendParams = BlendParams_import
    FoVPerspectiveCameras = FoVPerspectiveCameras_import
    MeshRasterizer = MeshRasterizer_import
    MeshRenderer = MeshRenderer_import
    RasterizationSettings = RasterizationSettings_import
    SoftSilhouetteShader = SoftSilhouetteShader_import
    look_at_view_transform = look_at_view_transform_import
    Meshes = Meshes_import
    ico_sphere = ico_sphere_import
    tqdm = tqdm_import

COW_BASE_URL = "https://raw.githubusercontent.com/facebookresearch/pytorch3d/main/docs/tutorials/data/cow_mesh"
COW_FILES = ("cow.obj", "cow.mtl", "cow_texture.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Differentiable cow mesh fitting from silhouettes.")
    parser.add_argument("--target-obj", type=Path, default=None, help="Path to target cow OBJ.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/cow_mesh"), help="Directory for downloaded cow data.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="Directory for rendered outputs.")
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device. Use 'auto' to select cuda:0 when available, otherwise cpu.",
    )
    parser.add_argument("--iters", type=int, default=1000, help="Optimization iterations.")
    parser.add_argument("--num-views", type=int, default=20, help="Number of training camera views.")
    parser.add_argument("--image-size", type=int, default=256, help="Square render size in pixels.")
    parser.add_argument("--sphere-level", type=int, default=4, help="Ico-sphere subdivision level.")
    parser.add_argument("--dist", type=float, default=2.7, help="Camera distance.")
    parser.add_argument("--elev", type=float, default=15.0, help="Camera elevation in degrees.")
    parser.add_argument("--lr", type=float, default=1e-2, help="Adam learning rate.")
    parser.add_argument("--sigma", type=float, default=1e-4, help="Soft silhouette sigmoid blur radius.")
    parser.add_argument("--gamma", type=float, default=1e-4, help="Soft silhouette blending gamma.")
    parser.add_argument("--w-lap", type=float, default=0.08, help="Laplacian smoothing weight.")
    parser.add_argument("--w-edge", type=float, default=0.5, help="Edge length penalty weight.")
    parser.add_argument("--w-normal", type=float, default=0.01, help="Normal consistency weight.")
    parser.add_argument("--save-every", type=int, default=100, help="Save a progress image every N iterations.")
    return parser.parse_args()


def download_default_cow(data_dir: Path) -> Path:
    """Download the PyTorch3D tutorial cow assets if no target OBJ is supplied."""

    data_dir.mkdir(parents=True, exist_ok=True)
    for filename in COW_FILES:
        destination = data_dir / filename
        if not destination.exists():
            urllib.request.urlretrieve(f"{COW_BASE_URL}/{filename}", destination)
    return data_dir / "cow.obj"


def normalize_mesh(mesh: Meshes) -> Meshes:
    """Center a mesh at the origin and scale it into a unit-radius box."""

    verts = mesh.verts_packed()
    center = verts.mean(dim=0)
    scale = (verts - center).abs().max()
    normalized = mesh.offset_verts(-center).scale_verts(1.0 / scale)
    # Match the source sphere scale so the first render overlaps the target well.
    return normalized.scale_verts(0.8)


def make_cameras(num_views: int, dist: float, elev: float, device: torch.device) -> FoVPerspectiveCameras:
    """Create azimuth-spaced cameras around the object."""

    azim = torch.linspace(-180.0, 180.0, num_views + 1, device=device)[:-1]
    elevs = torch.full_like(azim, elev)
    distances = torch.full_like(azim, dist)
    rotation, translation = look_at_view_transform(dist=distances, elev=elevs, azim=azim)
    return FoVPerspectiveCameras(device=device, R=rotation, T=translation)


def make_silhouette_renderer(
    cameras: FoVPerspectiveCameras,
    image_size: int,
    sigma: float,
    gamma: float,
) -> MeshRenderer:
    """Construct a PyTorch3D soft silhouette renderer."""

    raster_settings = RasterizationSettings(
        image_size=image_size,
        blur_radius=math.log(1.0 / 1e-4 - 1.0) * sigma,
        faces_per_pixel=50,
    )
    blend_params = BlendParams(sigma=sigma, gamma=gamma)
    return MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
        shader=SoftSilhouetteShader(blend_params=blend_params),
    )


def render_alpha(renderer: MeshRenderer, mesh: Meshes, cameras: FoVPerspectiveCameras) -> torch.Tensor:
    """Render alpha-channel silhouettes for all cameras."""

    batched_mesh = mesh.extend(len(cameras.R))
    return renderer(batched_mesh, cameras=cameras)[..., 3]


def save_silhouette_grid(images: torch.Tensor, path: Path, cols: int = 5, title: str | None = None) -> None:
    """Save a grid of silhouette tensors with values in [0, 1]."""

    images = images.detach().cpu().clamp(0, 1)
    rows = math.ceil(images.shape[0] / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(2.2 * cols, 2.2 * rows))
    axes_iter: Iterable[plt.Axes] = axes.flat if hasattr(axes, "flat") else (axes,)
    for axis, index in zip(axes_iter, range(rows * cols)):
        axis.axis("off")
        if index < images.shape[0]:
            axis.imshow(images[index], cmap="gray", vmin=0, vmax=1)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_progress_pair(target: torch.Tensor, prediction: torch.Tensor, path: Path, max_views: int = 5) -> None:
    """Save target/prediction/error rows for a small set of views."""

    count = min(max_views, target.shape[0])
    error = (target[:count] - prediction[:count]).abs()
    rows = [("target", target[:count]), ("prediction", prediction[:count]), ("abs error", error)]
    fig, axes = plt.subplots(len(rows), count, figsize=(2.3 * count, 6.6))
    for row_index, (label, row_images) in enumerate(rows):
        for col_index in range(count):
            axis = axes[row_index, col_index] if count > 1 else axes[row_index]
            axis.imshow(row_images[col_index].detach().cpu(), cmap="gray", vmin=0, vmax=1)
            axis.axis("off")
            if col_index == 0:
                axis.set_ylabel(label)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_loss_curve(history: dict[str, list[float]], path: Path) -> None:
    """Plot the optimization loss components."""

    fig, axis = plt.subplots(figsize=(8, 5))
    for key, values in history.items():
        axis.plot(values, label=key)
    axis.set_xlabel("iteration")
    axis.set_ylabel("loss")
    axis.set_yscale("log")
    axis.grid(True, alpha=0.3)
    axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_mesh(mesh: Meshes, path: Path) -> None:
    """Write a single-mesh PyTorch3D Meshes object as OBJ."""

    verts = mesh.verts_packed().detach().cpu()
    faces = mesh.faces_packed().detach().cpu()
    save_obj(str(path), verts, faces)


def main() -> None:
    args = parse_args()
    load_runtime_dependencies()
    selected_device = "cuda:0" if args.device == "auto" and torch.cuda.is_available() else args.device
    if selected_device == "auto":
        selected_device = "cpu"
    device = torch.device(selected_device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    target_obj = args.target_obj if args.target_obj is not None else download_default_cow(args.data_dir)
    target_mesh = normalize_mesh(load_objs_as_meshes([str(target_obj)], device=device))
    cameras = make_cameras(args.num_views, args.dist, args.elev, device)
    renderer = make_silhouette_renderer(cameras, args.image_size, args.sigma, args.gamma)

    with torch.no_grad():
        target_silhouettes = render_alpha(renderer, target_mesh, cameras)
    save_silhouette_grid(target_silhouettes, args.output_dir / "target_silhouettes.png", title="target cow silhouettes")

    source_mesh = ico_sphere(args.sphere_level, device=device).scale_verts(0.8)
    deform_verts = torch.zeros_like(source_mesh.verts_packed(), requires_grad=True)
    optimizer = torch.optim.Adam([deform_verts], lr=args.lr)

    history: dict[str, list[float]] = {"total": [], "silhouette": [], "laplacian": [], "edge": [], "normal": []}
    progress = tqdm(range(1, args.iters + 1), desc="optimizing")
    for iteration in progress:
        optimizer.zero_grad()
        deformed_mesh = source_mesh.offset_verts(deform_verts)
        predicted_silhouettes = render_alpha(renderer, deformed_mesh, cameras)

        loss_silhouette = F.mse_loss(predicted_silhouettes, target_silhouettes)
        loss_laplacian = mesh_laplacian_smoothing(deformed_mesh, method="uniform")
        loss_edge = mesh_edge_loss(deformed_mesh)
        loss_normal = mesh_normal_consistency(deformed_mesh)
        loss = (
            loss_silhouette
            + args.w_lap * loss_laplacian
            + args.w_edge * loss_edge
            + args.w_normal * loss_normal
        )
        loss.backward()
        optimizer.step()

        history["total"].append(float(loss.detach().cpu()))
        history["silhouette"].append(float(loss_silhouette.detach().cpu()))
        history["laplacian"].append(float(loss_laplacian.detach().cpu()))
        history["edge"].append(float(loss_edge.detach().cpu()))
        history["normal"].append(float(loss_normal.detach().cpu()))
        progress.set_postfix(total=history["total"][-1], silhouette=history["silhouette"][-1])

        should_save = iteration == 1 or iteration % args.save_every == 0 or iteration == args.iters
        if should_save:
            save_progress_pair(
                target_silhouettes,
                predicted_silhouettes,
                args.output_dir / f"progress_{iteration:04d}.png",
            )

    final_mesh = source_mesh.offset_verts(deform_verts)
    save_mesh(final_mesh, args.output_dir / "deformed_mesh.obj")
    save_loss_curve(history, args.output_dir / "loss_curve.png")
    print(f"Saved outputs to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
