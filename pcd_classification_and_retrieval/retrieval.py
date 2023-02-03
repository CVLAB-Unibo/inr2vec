from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import numpy as np
import torch
from sklearn.neighbors import KDTree
from torch import Tensor
from torch.utils.data import Dataset


def get_recalls(gallery: Tensor, labels_gallery: Tensor, kk: List[int]) -> Dict[int, float]:
    max_nn = max(kk)
    recalls = {idx: 0.0 for idx in kk}
    targets = labels_gallery.cpu().numpy()
    gallery = gallery.cpu().numpy()
    tree = KDTree(gallery)

    for query, label_query in zip(gallery, targets):
        with torch.no_grad():
            query = np.expand_dims(query, 0)
            _, indices_matched = tree.query(query, k=max_nn + 1)
            indices_matched = indices_matched[0]

            for k in kk:
                indices_matched_temp = indices_matched[1 : k + 1]
                classes_matched = targets[indices_matched_temp]
                recalls[k] += np.count_nonzero(classes_matched == label_query) > 0

    for key, value in recalls.items():
        recalls[key] = value / (1.0 * len(gallery))

    return recalls


class InrEmbeddingDataset(Dataset):
    def __init__(self, root: Path, split: str) -> None:
        super().__init__()

        self.root = root / split
        self.item_paths = sorted(self.root.glob("*.h5"), key=lambda x: int(x.stem))

    def __len__(self) -> int:
        return len(self.item_paths)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        with h5py.File(self.item_paths[index], "r") as f:
            embedding = np.array(f.get("embedding"))
            embedding = torch.from_numpy(embedding)
            class_id = np.array(f.get("class_id"))
            class_id = torch.from_numpy(class_id).long()

        return embedding, class_id


def main() -> None:
    inrs_embeddings_root = Path("/path/to/inrs/embeddings")
    test_dset = InrEmbeddingDataset(inrs_embeddings_root, "test")

    embeddings = []
    labels = []

    for i in range(len(test_dset)):
        embedding, label = test_dset[i]
        embeddings.append(embedding)
        labels.append(label)

    embeddings = torch.stack(embeddings)
    labels = torch.stack(labels)

    recalls = get_recalls(embeddings, labels, [1, 5, 10])
    for key, value in recalls.items():
        print(f"Recall@{key} : {100. * value:.2f}%")


if __name__ == "__main__":
    main()
