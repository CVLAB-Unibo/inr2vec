import sys

sys.path.append("..")

from pathlib import Path
from random import randint
from typing import Any, Dict

import numpy as np
import torch
from hesiod import get_out_dir, hcfg, hmain
from pycarus.geometry.mesh import get_o3d_mesh_from_tensors, marching_cubes
from torch import Tensor

from models.idecoder import ImplicitDecoder

import open3d as o3d  # isort: skip

latent_gan_embeddings_path = Path(sys.argv[2])


@hmain(base_cfg_dir="cfg/bases", run_cfg_file=sys.argv[1], parse_cmd_line=False)
def main() -> None:
    ckpt_path = get_out_dir() / "ckpts/best.pt"
    ckpt = torch.load(ckpt_path)

    embeddings = np.load(latent_gan_embeddings_path)["embeddings"]
    embeddings = torch.from_numpy(embeddings)

    encoder_cfg = hcfg("encoder", Dict[str, Any])
    decoder_cfg = hcfg("decoder", Dict[str, Any])
    decoder = ImplicitDecoder(
        encoder_cfg["embedding_dim"],
        decoder_cfg["input_dim"],
        decoder_cfg["hidden_dim"],
        decoder_cfg["num_hidden_layers_before_skip"],
        decoder_cfg["num_hidden_layers_after_skip"],
        decoder_cfg["out_dim"],
    )
    decoder.load_state_dict(ckpt["decoder"])
    decoder = decoder.cuda()
    decoder.eval()

    while True:
        idx = randint(0, embeddings.shape[0])
        emb = embeddings[idx].unsqueeze(0).cuda()

        def levelset_func(c: Tensor) -> Tensor:
            pred = decoder(emb, c.unsqueeze(0))
            pred = torch.sigmoid(pred.squeeze(0))
            pred *= 0.2
            pred -= 0.1
            return pred

        pred_v, pred_t = marching_cubes(levelset_func, (-1, 1), 128)
        output_o3d = get_o3d_mesh_from_tensors(pred_v, pred_t)

        o3d.visualization.draw_geometries([output_o3d])


if __name__ == "__main__":
    main()
