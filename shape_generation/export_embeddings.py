from pathlib import Path

import h5py
import numpy as np
from pycarus.utils import progress_bar


def main() -> None:
    inr_embeddings_root = Path("/path/to/inr/embeddings")
    out_root = Path("/path/out")
    num_classes = 10

    embeddings_paths = inr_embeddings_root.glob("*.h5")

    embeddings = {}
    for cls in range(num_classes):
        embeddings[cls] = []

    for path in progress_bar(embeddings_paths, "Saving"):
        with h5py.File(path, "r") as f:
            embedding = np.array(f.get("embedding"))
            class_id = np.array(f.get("class_id")).item()
            embeddings[class_id].append(embedding)

    for cls, embeddings in embeddings.items():
        path_out = out_root / f"embeddings_{cls}.npz"
        embeddings = np.stack(embeddings)
        np.savez_compressed(path_out, embeddings=embeddings)


if __name__ == "__main__":
    main()
