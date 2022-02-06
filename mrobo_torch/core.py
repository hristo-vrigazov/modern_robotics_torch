import numpy as np
import modern_robotics as mr
import torch


def VecToso3(omg):
    # converts to 3.30
    kwargs = {
        'dtype': omg.dtype,
        'device': omg.device
    }
    if len(omg.shape) == 1:
        return torch.tensor([[0, -omg[2], omg[1]],
                             [omg[2], 0, -omg[0]],
                             [-omg[1], omg[0], 0]], **kwargs)
    assert len(omg.shape) == 2
    res = torch.zeros(len(omg), 3, 3, **kwargs)

    res[:, 1, 2] = -omg[:, 0]
    res[:, 2, 1] = omg[:, 0]
    res[:, 2, 0] = -omg[:, 1]
    res[:, 0, 2] = omg[:, 1]
    res[:, 0, 1] = -omg[:, 2]
    res[:, 1, 0] = omg[:, 2]
    return res


def VecTose3(V):
    # Converts 3.70 to 3.71
    kwargs = {
        'dtype': V.dtype,
        'device': V.device
    }
    if len(V.shape) == 1:
        so3_vec = VecToso3(V[:3])
        next_v = V[3:].unsqueeze(dim=1)
        omega_and_v = torch.cat((so3_vec, next_v), dim=1)
        row = torch.zeros(1, 4, **kwargs)
        res = torch.cat((omega_and_v, row), dim=0)
        return res

    res = torch.zeros(len(V), 4, 4, **kwargs)
    res[:, :3, :3] = VecToso3(V[:, :3])
    res[:, :3, 3] = V[:, 3:]
    return res


def so3ToVec(so3mat):
    # Inverse conversion from 3.30
    if len(so3mat.shape) == 2:
        return torch.tensor([so3mat[2][1],
                             so3mat[0][2],
                             so3mat[1][0]],
                            dtype=so3mat.dtype,
                            device=so3mat.device)
    return torch.stack((so3mat[:, 2, 1],
                        so3mat[:, 0, 2],
                        so3mat[:, 1, 0]),
                       dim=1)


def MatrixExp3(so3mat, eps=1e-6):
    omgtheta = so3ToVec(so3mat)
    kwargs = {
        'dtype': omgtheta.dtype,
        'device': omgtheta.device
    }
    if len(omgtheta.shape) == 1:
        if omgtheta.norm().item() < eps:
            return torch.eye(3, **kwargs)

        theta = omgtheta.norm()
        omgmat = so3mat / theta
        initial = torch.eye(3, **kwargs)
        sin_term = torch.sin(theta) * omgmat
        cos_term = (1 - torch.cos(theta)) * torch.mm(omgmat, omgmat)
        return initial + sin_term + cos_term

    res = torch.zeros(len(so3mat), 3, 3, **kwargs)
    theta = omgtheta.norm(dim=1)
    res[theta < eps] = torch.eye(3, **kwargs)
    base_mask = ~(theta < eps)
    theta_u = theta.unsqueeze(dim=-1).unsqueeze(dim=-1)
    omgmat = so3mat[base_mask] / theta_u[base_mask]
    res[base_mask] = torch.eye(3, **kwargs)
    res[base_mask] += torch.sin(theta_u[base_mask]) * omgmat
    cos_term = (1 - torch.cos(theta_u[base_mask]))
    omg_term = torch.matmul(omgmat, omgmat)
    res[base_mask] += cos_term * omg_term
    return res


def MatrixExp6(se3mat, eps=1e-6):
    kwargs = {
        'dtype': se3mat.dtype,
        'device': se3mat.device
    }
    if len(se3mat.shape) == 2:
        omgtheta = so3ToVec(se3mat[:3, :3])
        if omgtheta.norm().item() < eps:
            # 3.89
            I = torch.eye(3, **kwargs)
            v = se3mat[:3, 3:]
            res = torch.cat((I, v), dim=1)
            last_row = torch.tensor([[0, 0, 0, 1]], **kwargs)
            return torch.cat((res, last_row), dim=0)
        # 3.88
        theta = omgtheta.norm()
        omgmat = se3mat[:3, :3] / theta
        exp3_mat = MatrixExp3(se3mat[:3, :3])

        initial = torch.eye(3, **kwargs) * theta
        cos_term = (1 - torch.cos(theta)) * omgmat
        omg_term = torch.mm(omgmat, omgmat)
        sin_term = (theta - torch.sin(theta)) * omg_term
        composite_term = initial + cos_term + sin_term
        v = se3mat[:3, 3]
        res = torch.mv(composite_term, v) / theta
        res = res.unsqueeze(1)
        res = torch.cat((exp3_mat, res), dim=1)
        last_row = torch.tensor([[0, 0, 0, 1]], **kwargs)
        res = torch.cat((res, last_row), dim=0)
        return res

    omgtheta = so3ToVec(se3mat[:, :3, :3])

    res = torch.zeros(len(se3mat), 4, 4, **kwargs)
    theta = omgtheta.norm(dim=1)

    # 3.89
    res[theta < eps, :3, :3] = torch.eye(3, **kwargs)
    res[theta < eps, :3, 3] = se3mat[theta < eps, :3, 3]
    res[:, 3, 3] = 1.

    # 3.88
    mask = ~(theta < eps)
    theta_u = theta[mask].unsqueeze(dim=-1).unsqueeze(dim=-1)
    omgmat = se3mat[mask, :3, :3] / theta_u
    exp3_mat = MatrixExp3(se3mat[mask, :3, :3])

    eye = torch.eye(3, **kwargs)
    eye = eye.unsqueeze(dim=0)
    initial = eye * theta_u
    cos_term = (1 - torch.cos(theta_u)) * omgmat
    omg_term = torch.matmul(omgmat, omgmat)
    sin_term = (theta_u - torch.sin(theta_u)) * omg_term
    composite_term = initial + cos_term + sin_term
    v = se3mat[mask, :3, 3].unsqueeze(dim=-1)
    mul_term = torch.matmul(composite_term, v) / theta_u

    res[mask, :3, :3] = exp3_mat
    res[mask, :3, 3:] = mul_term
    return res


class TorchKinematics:

    def __init__(self, M, Slist, device='cpu', eomg=1e-3, ev=1e-3):
        self.eomg = eomg
        self.ev = ev

        self.device = torch.device(device)
        self.M = torch.tensor(M, dtype=float, device=self.device)
        self.Slist = torch.tensor(Slist, dtype=float, device=self.device)

    def forward(self, thetalist):
        thetalist = torch.as_tensor(thetalist,
                                    device=self.device,
                                    dtype=float)
        is_batch = not (len(thetalist.shape) == 1)
        M = self.M
        Slist = self.Slist
        T = M.clone()
        n = len(thetalist)
        if is_batch:
            T = torch.repeat_interleave(T.unsqueeze(dim=0),
                                        len(thetalist),
                                        dim=0)
            n = thetalist.shape[1]
        for i in range(n - 1, -1, -1):
            if is_batch:
                theta_for_joint = thetalist[:, i:i + 1]
                Slist_for_joint = Slist[:, i].unsqueeze(dim=0)
            else:
                theta_for_joint = thetalist[i]
                Slist_for_joint = Slist[:, i]
            s_theta = Slist_for_joint * theta_for_joint
            se3 = VecTose3(s_theta)
            exp_res = MatrixExp6(se3)
            T = torch.matmul(exp_res, T)
        return T

    def inverse(self, goal, initial_guess=None):
        T = goal
        thetalist0 = initial_guess if initial_guess is not None else np.zeros(5)
        M = self.M.cpu().numpy()
        Slist = self.Slist.cpu().numpy()
        eomg = self.eomg
        ev = self.ev
        thetalist = np.array(thetalist0).copy()
        i = 0
        maxiterations = 20
        Vs, err = self.compute_Vs(M, Slist, T, eomg, ev, thetalist)
        while err and i < maxiterations:
            thetalist = thetalist + np.dot(np.linalg.pinv(mr.JacobianSpace(Slist, thetalist)), Vs)
            i = i + 1
            Vs, err = self.compute_Vs(M, Slist, T, eomg, ev, thetalist)
        return thetalist, not err

    def compute_Vs(self, M, Slist, T, eomg, ev, thetalist):
        Tsb = mr.FKinSpace(M, Slist, thetalist)
        Vs = np.dot(mr.Adjoint(Tsb), mr.se3ToVec(mr.MatrixLog6(np.dot(mr.TransInv(Tsb), T))))
        err = np.linalg.norm([Vs[0], Vs[1], Vs[2]]) > eomg or np.linalg.norm([Vs[3], Vs[4], Vs[5]]) > ev
        return Vs, err


def get_px150_M_Slist():
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
    return M, Slist
