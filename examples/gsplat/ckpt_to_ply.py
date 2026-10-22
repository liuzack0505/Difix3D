"""Convert a gsplat .pt checkpoint to a standard 3DGS .ply file.

Writes the same layout simple_trainer_difix3d.py reads with --ckpt, so the
result can be loaded back for fine-tuning or opened in 3DGS viewers.

Example:
    python examples/gsplat/ckpt_to_ply.py \
        data/test_0923_NT_less_01_result/ckpts/ckpt_29999_rank0.pt
"""

import argparse
import os

import numpy as np
import torch
from plyfile import PlyData, PlyElement


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("ckpt", help="Input gsplat checkpoint (.pt).")
    parser.add_argument("--output", help="Output .ply path (default: next to the checkpoint).")
    args = parser.parse_args()

    splats = torch.load(args.ckpt, map_location="cpu", weights_only=True)["splats"]
    if "sh0" not in splats:
        raise ValueError("Only SH-color checkpoints are supported (app_opt checkpoints have no sh0).")
    n = len(splats["means"])

    # Values stay in optimizer space (log scales, logit opacities), matching the 3DGS convention.
    # Standard 3DGS stores SH coefficients channel-first; gsplat uses [N, K, 3].
    f_dc = splats["sh0"].transpose(1, 2).reshape(n, -1)
    f_rest = splats["shN"].transpose(1, 2).reshape(n, -1)
    names = (
        ["x", "y", "z", "nx", "ny", "nz"]
        + [f"f_dc_{i}" for i in range(f_dc.shape[1])]
        + [f"f_rest_{i}" for i in range(f_rest.shape[1])]
        + ["opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    )
    # gsplat normalizes quaternions at render time, so trained ones drift off unit length;
    # viewers such as SuperSplat expect unit quaternions.
    quats = torch.nn.functional.normalize(splats["quats"], dim=-1)
    data = torch.cat([
        splats["means"], torch.zeros(n, 3), f_dc, f_rest,
        splats["opacities"].reshape(n, 1), splats["scales"], quats,
    ], dim=1).float().numpy()
    vertices = np.empty(n, dtype=[(name, "f4") for name in names])
    for i, name in enumerate(names):
        vertices[name] = data[:, i]

    output = args.output or os.path.splitext(args.ckpt)[0] + ".ply"
    PlyData([PlyElement.describe(vertices, "vertex")]).write(output)
    print(f"Wrote {output}: {n} Gaussians, SH degree {int(np.sqrt(splats['shN'].shape[1] + 1)) - 1}")


if __name__ == "__main__":
    main()
