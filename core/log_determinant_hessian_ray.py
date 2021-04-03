import numpy as np
import torch
from torch.distributions.kl import kl_divergence
from core.models import detach_distribution
from torch.nn.utils.convert_parameters import parameters_to_vector, vector_to_parameters
from hessian import hessian
import ray


def _compute_kl(policy_net, states):
    pi = policy_net(states)
    pi_detach = detach_distribution(pi)
    kl = torch.mean(kl_divergence(pi_detach, pi))
    return kl


@ray.remote(num_gpus=1)
def _compute_log_determinant(pid, policy_net, states, matrix_dim, damping=1e-2, device='cpu'):
    kl = _compute_kl(policy_net, states)
    Hmatrix = hessian(kl, policy_net.parameters()) + torch.eye(matrix_dim).to(device).double()*damping
    l = torch.cholesky(Hmatrix)
    log_det_exact = 2 * l.diagonal(dim1=-2, dim2=-1).log().sum(-1)
    return pid, log_det_exact


def compute_log_determinant(policy_net, states_list, matrix_dim, damping=1e-2, num_trace=40, cheby_degree=100, eigen_amp=1, device='cpu'):
    agent_count = len(states_list)
    result_ids = []
    log_determinant = [None]*agent_count
    policy_net = policy_net.to(device)

    for states, index in zip(states_list, range(agent_count)):
        states = states.to(device)
        result_id = _compute_log_determinant.remote(
            pid=index,
            policy_net=policy_net,
            states=states,
            matrix_dim=matrix_dim,
            damping=damping,
            device=device
        )
        result_ids.append(result_id)

    for result_id in result_ids:
        pid, log_det = ray.get(result_id)
        log_determinant[pid] = (log_det.to('cpu'))

    policy_net = policy_net.to('cpu')

    return log_determinant