from pathlib import Path

import h5py
import numpy as np
from pycarus.geometry.mesh import compute_sdf_from_mesh, get_o3d_mesh_from_tensors
from pycarus.utils import progress_bar


def main() -> None:
    inrs_dir = Path("/path/to/inrs/train")
    out_dir = Path("/path/to/inrs/train_sdf")
    out_dir.mkdir(parents=True, exist_ok=True)

    num_queries_on_surface = 100_000
    stds = [0.003, 0.01, 0.1]
    num_points_per_std = [250_000, 200_000, 25_000, 25_000]

    inrs_paths = sorted(inrs_dir.glob("*.h5"), key=lambda x: int(x.stem))

    for p in progress_bar(inrs_paths):
        with h5py.File(p, "r") as f:
            v = np.array(f.get("vertices"))
            num_v = np.array(f.get("num_vertices"))
            t = np.array(f.get("triangles"))
            num_t = np.array(f.get("num_triangles"))
            params = np.array(f.get("params"))
            class_id = np.array(f.get("class_id"))

        mesh_o3d = get_o3d_mesh_from_tensors(v[:num_v], t[:num_t])

        coords, labels = compute_sdf_from_mesh(
            mesh_o3d,
            num_queries_on_surface=num_queries_on_surface,
            queries_stds=stds,
            num_queries_per_std=num_points_per_std,
        )

        out_path = out_dir / p.name

        with h5py.File(out_path, "w") as f:
            f.create_dataset("vertices", data=v)
            f.create_dataset("num_vertices", data=num_v)
            f.create_dataset("triangles", data=t)
            f.create_dataset("num_triangles", data=num_t)
            f.create_dataset("params", data=params)
            f.create_dataset("class_id", data=class_id)
            f.create_dataset("coords", data=coords.detach().cpu().numpy())
            f.create_dataset("labels", data=labels.detach().cpu().numpy())


if __name__ == "__main__":
    main()
