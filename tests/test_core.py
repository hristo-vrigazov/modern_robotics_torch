import modern_robotics as mr
import numpy as np
import pytest
import torch

from mrobo_torch import core
from mrobo_torch.core import TorchKinematics


def test_MatrixExp3_matches():
    so3_v = torch.rand(2 ** 10, 3, 3)
    all_r = []
    for i in range(len(so3_v)):
        all_r.append(mr.MatrixExp3(so3_v[i].numpy()))
    final_res = torch.tensor(np.stack(all_r, axis=0)).float()

    actual_res = core.MatrixExp3(so3_v)
    torch.allclose(final_res, actual_res.cpu())


@pytest.mark.parametrize("device", [
    "cpu",
    "cuda:0"
])
def test_forward_kinematics_matches(device):
    M = np.array([
        [1., 0., 0., 0.358575],
        [0., 1., 0., 0.],
        [0., 0., 1., 0.25391],
        [0., 0., 0., 1.]])
    Slist = np.array([
        [0., 0., 0., 0., 1.],
        [0., 1., -1., -1., 0.],
        [1., 0., 0., 0., 0.],
        [0., -0.10391, 0.25391, 0.25391, 0.],
        [0., 0., 0., 0., 0.25391],
        [0., 0., -0.05, -0.2, 0.]
    ])

    kinematics = TorchKinematics(M, Slist, device=device)
    thetalist = torch.rand(2 ** 13, 5, device=device)
    actual = kinematics.forward(thetalist)

    all_r = [mr.FKinSpace(M, Slist, t.cpu().numpy()) for t in thetalist]
    expected = torch.tensor(np.stack(all_r, axis=0)).to(device)

    assert torch.allclose(expected, actual)
