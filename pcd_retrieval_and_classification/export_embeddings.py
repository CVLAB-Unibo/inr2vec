import sys

sys.path.append("..")

from pathlib import Path
from typing import Any, Dict, Tuple

import h5py
import numpy as np
import torch
from hesiod import hcfg, hmain
from pycarus.learning.models.siren import SIREN
from pycarus.utils import progress_bar
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from models.encoder import Encoder
from utils import get_mlp_params_as_matrix


class InrDataset(Dataset):
    def __init__(self, inrs_root: Path, split: str, sample_sd: Dict[str, Any]) -> None:
        super().__init__()

        self.inrs_root = inrs_root / split
        self.mlps_paths = sorted(self.inrs_root.glob("*.h5"), key=lambda x: int(x.stem))
        self.sample_sd = sample_sd

    def __len__(self) -> int:
        return len(self.mlps_paths)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor, Tensor]:
        with h5py.File(self.mlps_paths[index], "r") as f:
            pcd = torch.from_numpy(np.array(f.get("pcd")))
            params = np.array(f.get("params"))
            params = torch.from_numpy(params).float()
            matrix = get_mlp_params_as_matrix(params, self.sample_sd)
            class_id = torch.from_numpy(np.array(f.get("class_id"))).long()

        return pcd, matrix, class_id


@hmain(
    base_cfg_dir="cfg/bases",
    template_cfg_file="cfg/export_embeddings.yaml",
    create_out_dir=False,
)
def main() -> None:
    inrs_root = Path(hcfg("inrs_root", str))

    mlp_hdim = hcfg("mlp.hidden_dim", int)
    num_hidden_layers = hcfg("mlp.num_hidden_layers", int)
    mlp = SIREN(3, mlp_hdim, num_hidden_layers, 1)
    sample_sd = mlp.state_dict()

    train_split = hcfg("train_split", str)
    train_dset = InrDataset(inrs_root, train_split, sample_sd)
    train_loader = DataLoader(train_dset, batch_size=1, num_workers=0, shuffle=False)

    val_split = hcfg("val_split", str)
    val_dset = InrDataset(inrs_root, val_split, sample_sd)
    val_loader = DataLoader(val_dset, batch_size=1, num_workers=0, shuffle=False)

    test_split = hcfg("test_split", str)
    test_dset = InrDataset(inrs_root, test_split, sample_sd)
    test_loader = DataLoader(test_dset, batch_size=1, num_workers=0, shuffle=False)

    encoder_cfg = hcfg("encoder", Dict[str, Any])
    encoder = Encoder(
        mlp_hdim,
        encoder_cfg["hidden_dims"],
        encoder_cfg["embedding_dim"],
    )
    ckpt = torch.load(hcfg("ckpt_path", str))
    encoder.load_state_dict(ckpt["encoder"])
    encoder = encoder.cuda()
    encoder.eval()

    loaders = [train_loader, val_loader, test_loader]
    splits = [train_split, val_split, test_split]

    for loader, split in zip(loaders, splits):
        idx = 0

        for batch in progress_bar(loader, f"{split}"):
            pcds, matrices, class_ids = batch
            matrices = matrices.cuda()

            with torch.no_grad():
                embedding = encoder(matrices)

            h5_path = Path(hcfg("out_root", str)) / Path(f"{split}") / f"{idx}.h5"
            h5_path.parent.mkdir(parents=True, exist_ok=True)

            with h5py.File(h5_path, "w") as f:
                f.create_dataset("pcd", data=pcds[0].detach().cpu().numpy())
                f.create_dataset("embedding", data=embedding[0].detach().cpu().numpy())
                f.create_dataset("class_id", data=class_ids[0].detach().cpu().numpy())

            idx += 1


if __name__ == "__main__":
    main()
