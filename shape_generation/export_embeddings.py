import sys
from pathlib import Path

import h5py
import numpy as np
from pycarus.utils import progress_bar


def main() -> None:
    inr_embeddings_root = Path(sys.argv[1])
    out_root = Path(sys.argv[2])
    out_root.mkdir(parents=True, exist_ok=True)
    num_classes = 40

    embeddings_paths = list(inr_embeddings_root.glob("*.h5"))

    embeddings = {}
    for cls in range(num_classes):
        embeddings[cls] = []

    for path in progress_bar(embeddings_paths, "Extracting embeddings"):
        with h5py.File(path, "r") as f:
            embedding = np.array(f.get("embedding"))
            class_id = np.array(f.get("class_id")).item()
            embeddings[class_id].append(embedding)

    for class_id in progress_bar(range(num_classes), "Saving embeddings"):
        path_out = out_root / f"embeddings_{class_id}.npz"
        embeddings = np.stack(embeddings[class_id])
        np.savez_compressed(path_out, embeddings=embeddings)


if __name__ == "__main__":
    main()
