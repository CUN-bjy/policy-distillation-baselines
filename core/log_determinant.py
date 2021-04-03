import numpy as np
import torch
from torch.distributions.kl import kl_divergence
from core.models import detach_distribution


def _compute_log_determinant(mvp, num_trace, cheby_degree, l_min, l_max, matrix_dim):
    a = l_min+l_max
    delta = l_min / a
    mvp_scale = lambda x: mvp(x)/a
    # matrix = matrix/a
    f = lambda x: np.log(1 - x)
    g = lambda x: (.5 - delta)*x+.5
    ginv = lambda x: x/(.5 - delta)
    h = lambda x: f(g(x))
    # print("computing cheby poly weights")
    cheby_weights = chebyshev_poly_weights(h, cheby_degree)
    v = 2.0*(np.sign(np.random.randint(low=0, high=2, size=(matrix_dim, num_trace))))-1.0
    # print(np.shape(v))
    u = cheby_weights[0]*v
    if cheby_degree>1:
        w0 = v
        w1 = mvp_scale(v)
        w1 = ginv(w1)
        w1 = v/(1-2*delta) - w1
        u = cheby_weights[1]*w1 + cheby_weights[0]*w0
        for j in range(2, cheby_degree+1):
            # print(j)
            ww = mvp_scale(w1)
            ww = ginv(ww)
            ww = w1/(1-2*delta)-ww
            ww = 2*ww - w0
            u = cheby_weights[j]*ww + u
            w0 = w1
            w1 = ww

    ld = np.sum(np.sum(v*u))/num_trace + matrix_dim*np.log(a)

    return ld


def chebyshev_poly_weights(f, cheby_degree):
    x = np.cos(np.pi*(np.arange(cheby_degree+1)+0.5)/(cheby_degree+1))
    y = f(x)
    T = np.stack([np.zeros(cheby_degree+1), np.ones(cheby_degree+1)], axis=1)
    c = np.zeros(cheby_degree+1)
    c[0] = np.mean(y)
    a = 1
    for index in range(1, cheby_degree+1):
        T = np.stack([T[:, 1], a*x*T[:, 1] - T[:, 0]], axis=1)
        c[index] = np.dot(y, T[:, 1])*2/(cheby_degree+1)
        a = 2

    return c


def _fvsp(policy_net, states, damping=1e-2, device='cpu'):
    pi = policy_net(states)
    pi_detach = detach_distribution(pi)
    kl = torch.mean(kl_divergence(pi_detach, pi))
    grads = torch.autograd.grad(kl, policy_net.parameters(), create_graph=True, retain_graph=True)
    flat_grad_kl = torch.cat([grad.view(-1) for grad in grads])
    def __fvsp(vectors, damping=damping):
        vectors = torch.from_numpy(vectors).to(device)
        results = []
        vector_list = []
        if len(vectors.shape) > 1:
            for j in range(vectors.shape[1]):
                vector_list.append(torch.squeeze(vectors[:, j]))
        else:
            vector_list.append(vectors)
        for vector in vector_list:
            kl_v = (flat_grad_kl * vector).sum()
            grads = torch.autograd.grad(kl_v, policy_net.parameters(), retain_graph=True)
            flat_grad_grad_kl = torch.cat([grad.view(-1) for grad in grads])
            results.append((flat_grad_grad_kl + vector * damping))
        return np.squeeze(torch.stack(results, dim=1).cpu().numpy())

    return __fvsp


def estimate_largest_eigenvalue(mvp, matrix_dim, power_iteration_count=100):
    x = np.random.randn(matrix_dim)
    x = x / np.linalg.norm(x)
    # print(np.linalg.norm(x))
    # print(np.shape(x))
    for i in range(power_iteration_count):
        # print(i)
        x = mvp(x)
        # x = np.squeeze(x)
        # print(type(x))
        # print(np.shape(x))
        # print(np.linalg.norm(x))
        x = x/np.linalg.norm(x)
        # print(np.linalg.norm(x))

    return np.dot(x, mvp(x))


def compute_log_determinant(policy_net, states_list, matrix_dim, damping=1e-2, num_trace=40, cheby_degree = 100, eigen_amp = 1, device='cpu'):
    num_workers = len(states_list)
    result_ids = []
    log_determinant = []
    policy_net = policy_net.to(device)
    for states, index in zip(states_list, range(num_workers)):
        states = states.to(device)
        fvsp = _fvsp(policy_net, states, damping=damping, device=device)
        l_min = damping
        # print("estimating largest eigenvalue")
        l_max = estimate_largest_eigenvalue(fvsp, matrix_dim)*eigen_amp
        # print(l_max)
        # print("computing log determinant")
        log_det = _compute_log_determinant(fvsp, num_trace, cheby_degree, l_min, l_max, matrix_dim)
        log_determinant.append(log_det)
        print("\t Finishing {}/{} log dets.".format(index+1, num_workers))

    policy_net = policy_net.to('cpu')

    return log_determinant