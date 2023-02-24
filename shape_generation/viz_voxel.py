import sys

sys.path.append("..")

from pathlib import Path
from random import randint
from typing import Any, Dict, cast

import numpy as np
import torch
from einops import rearrange
from hesiod import get_out_dir, hcfg, hmain
from pycarus.geometry.mesh import get_o3d_mesh_from_tensors
from pycarus.geometry.pcd import normalize_pcd_into_unit_sphere, voxelize_pcd
from pytorch3d.ops import cubify
from torch import Tensor

from models.idecoder import ImplicitDecoder

import open3d as o3d  # isort: skip

latent_gan_embeddings_path = Path(sys.argv[2])


@hmain(base_cfg_dir="cfg/bases", run_cfg_file=sys.argv[1], parse_cmd_line=False)
def main() -> None:
    ckpt_path = get_out_dir() / "ckpts/best.pt"
    ckpt = torch.load(ckpt_path)

    vox_res = hcfg("vox_res", int)

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

    dummy_pcd = normalize_pcd_into_unit_sphere(torch.rand(2048, 3))
    _, centroids = voxelize_pcd(dummy_pcd, vox_res, -1, 1)
    centroids = centroids.unsqueeze(0)
    centroids = rearrange(centroids, "b r1 r2 r3 d -> b (r1 r2 r3) d")
    centroids = centroids.cuda()

    while True:
        idx = randint(0, embeddings.shape[0])
        emb = embeddings[idx].unsqueeze(0).cuda()

        with torch.no_grad():
            pred_vgrid = torch.sigmoid(decoder(emb, centroids))
            pred_vgrid = rearrange(pred_vgrid, "b (r1 r2 r3) -> b r1 r2 r3", r1=vox_res, r2=vox_res)

        pred_vgrid_cubified = cubify(pred_vgrid, 0.4, align="center")
        pred_v = cast(Tensor, pred_vgrid_cubified.verts_packed())
        pred_t = cast(Tensor, pred_vgrid_cubified.faces_packed())
        pred_vgrid_o3d = get_o3d_mesh_from_tensors(pred_v, pred_t)

        o3d.visualization.draw_geometries([pred_vgrid_o3d])


if __name__ == "__main__":
    main()
