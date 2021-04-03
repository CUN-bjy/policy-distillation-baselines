import multiprocessing as mp
from torch.distributions.kl import kl_divergence
from core.models import detach_distribution
import ray
from utils.utils import *

def cg(mvp, b, nsteps, residual_tol=1e-10):
    x = torch.zeros(b.size(), dtype=b.dtype)
    r = b.clone()
    p = b.clone()
    rdotr = torch.dot(r, r)
    for i in range(nsteps):
        _mvp = mvp(p)
        alpha = rdotr / torch.dot(p, _mvp)
        x += alpha * p
        r -= alpha * _mvp
        new_rdotr = torch.dot(r, r)
        betta = new_rdotr / rdotr
        p = r + betta * p
        rdotr = new_rdotr
        if rdotr < residual_tol:
            break
    return x

@ray.remote(num_gpus=1)
def conjugate_gradient(policy_net, states, pg, max_kl=1e-2, cg_damping=1e-2, cg_iter = 10, pid=None):
    torch.tensor(1).to('cuda')

    for param in policy_net.parameters():
        param.requires_grad = True

    def _fvp(states, damping=1e-2):
        def __fvp(vector, damping=damping):
            pi = policy_net(states)
            pi_detach = detach_distribution(pi)
            kl = torch.mean(kl_divergence(pi_detach, pi))

            grads = torch.autograd.grad(kl, policy_net.parameters(), create_graph=True)
            flat_grad_kl = torch.cat([grad.view(-1) for grad in grads])

            kl_v = (flat_grad_kl * vector).sum()
            grads = torch.autograd.grad(kl_v, policy_net.parameters())
            flat_grad_grad_kl = torch.cat([grad.view(-1) for grad in grads])

            return flat_grad_grad_kl + vector * damping

        return __fvp


    fvp = _fvp(states, damping=cg_damping)
    stepdir = cg(fvp, -pg, cg_iter)
    shs = 0.5 * (stepdir * fvp(stepdir)).sum(0, keepdim=True)
    lm = torch.sqrt(shs / max_kl)
    fullstep = stepdir / lm[0]

    return (pid, fullstep)


def conjugate_gradient_parallel(policy_net, states_list, pg, max_kl=1e-3, cg_damping=1e-2, cg_iter=10, num_parallel_workers=mp.cpu_count()):
    result_ids = []
    for states, index in zip(states_list, range(len(states_list))):
        result_ids.append(conjugate_gradient.remote(
            policy_net, states, pg, max_kl, cg_damping, cg_iter, index))

    stepdirs = [None] * len(states_list)
    for result_id in result_ids:
        pid, stepdir = ray.get(result_id)
        stepdirs[pid] = stepdir.numpy()

    return stepdirs

def local_conjugate_gradient_parallel_and_line_search(trpo_loss, compute_kl, policy_net, states_list, advantages_list, actions_list, pg_list, max_kl=1e-3, cg_damping=1e-2, cg_iter=10, num_parallel_workers=mp.cpu_count()):
    result_ids = []
    for states, pg, index in zip(states_list, pg_list, range(len(states_list))):
        result_ids.append(conjugate_gradient.remote(
            policy_net, states, torch.from_numpy(pg), max_kl, cg_damping, cg_iter, index))

    stepdirs = [None] * len(states_list)
    for result_id in result_ids:
        pid, stepdir = ray.get(result_id)
        stepdirs[pid] = stepdir

    prev_params = get_flat_params_from(policy_net)
    xnews = []
    for stepdir, states, actions, advantages in zip(stepdirs, states_list, actions_list, advantages_list):
        loss = trpo_loss(advantages, states, actions, prev_params, prev_params).detach().numpy()
        for (n_backtracks, stepfrac) in enumerate(0.5 ** np.arange(10)):
            xnew = prev_params + stepfrac * stepdir
            new_loss = trpo_loss(advantages, states, actions, prev_params, xnew).data.numpy()
            kl = compute_kl(states, prev_params, xnew).detach().numpy()
            # print(new_loss - fval, kl)
            if new_loss - loss < 0 and kl < max_kl:
                xnews.append(xnew.numpy())
                break

    return xnews

def conjugate_gradient_global(policy_net, states_list, pg, max_kl=1e-3, cg_damping=1e-2, cg_iter=10):
    states = torch.cat(states_list)
    for param in policy_net.parameters():
        param.requires_grad = True
    def _fvp(states, damping=1e-2):
        def __fvp(vector, damping=damping):
            pi = policy_net(states)
            pi_detach = detach_distribution(pi)
            kl = torch.mean(kl_divergence(pi_detach, pi))

            grads = torch.autograd.grad(kl, policy_net.parameters(), create_graph=True)
            flat_grad_kl = torch.cat([grad.view(-1) for grad in grads])

            kl_v = (flat_grad_kl * vector).sum()
            grads = torch.autograd.grad(kl_v, policy_net.parameters())
            flat_grad_grad_kl = torch.cat([grad.view(-1) for grad in grads])

            return flat_grad_grad_kl + vector * damping

        return __fvp

    fvp = _fvp(states, damping=cg_damping)
    stepdir = cg(fvp, -pg, cg_iter)
    shs = 0.5 * (stepdir * fvp(stepdir)).sum(0, keepdim=True)
    lm = torch.sqrt(shs / max_kl)
    fullstep = stepdir / lm[0]

    return fullstep