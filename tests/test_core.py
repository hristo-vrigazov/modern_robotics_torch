import modern_robotics as mr
import numpy as np
import torch

from mrobo_torch import core


def test_so3Vec_matches():
    so3_v = torch.rand(2 ** 10, 3, 3)
    all_r = []
    for i in range(len(so3_v)):
        all_r.append(mr.MatrixExp3(so3_v[i].numpy()))
    final_res = torch.tensor(np.stack(all_r, axis=0)).float()

    actual_res = core.MatrixExp3(so3_v)
    torch.allclose(final_res, actual_res.cpu())


def test_forward_kinematics_matches():
    pass
