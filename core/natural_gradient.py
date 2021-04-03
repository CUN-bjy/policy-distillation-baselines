import multiprocessing as mp
import torch
from torch.distributions.kl import kl_divergence
from core.models import detach_distribution
from torch.autograd import Variable
import math

def cg(mvp, b, nsteps, residual_tol=1e-10):
    x = torch.zeros(b.size())
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

def conjugate_gradient(policy_net, states_list, pg, max_kl=1e-3, cg_damping=1e-2, cg_iter = 10, queue=None, pid=None, num_agent=None):
    def _fvp(states, damping=1e-2):
        def __fvp(vector, damping=damping):
            pi = policy_net(Variable(states))
            pi_detach = detach_distribution(pi)
            kl = torch.mean(kl_divergence(pi_detach, pi))

            grads = torch.autograd.grad(kl, policy_net.parameters(), create_graph=True)
            flat_grad_kl = torch.cat([grad.view(-1) for grad in grads])

            kl_v = (flat_grad_kl * vector).sum()
            grads = torch.autograd.grad(kl_v, policy_net.parameters())
            flat_grad_grad_kl = torch.cat([grad.view(-1) for grad in grads])

            return flat_grad_grad_kl + vector * damping

        return __fvp

    for states, index in zip(states_list, range(len(states_list))):
        fvp = _fvp(states, damping=cg_damping)


        stepdir = cg(fvp, -pg, cg_iter)
        shs = 0.5 * (stepdir * fvp(stepdir)).sum(0, keepdim=True)
        lm = torch.sqrt(shs / max_kl)
        fullstep = stepdir / lm[0]

        if queue is not None:
            queue.put([pid * num_agent + index, fullstep])

    if pid is None:
        return fullstep


def conjugate_gradient_parallel(policy_net, states_list, pg, max_kl=1e-3, cg_damping=1e-2, cg_iter=10, num_parallel_workers=mp.cpu_count()):
    workers = []
    queue = mp.Queue()
    num_agents = len(states_list)
    process_agent_count = int(math.floor(num_agents / num_parallel_workers))
    for i in range(num_parallel_workers):
        worker_args = (
            policy_net, states_list[i * process_agent_count:(i + 1) * process_agent_count], pg, max_kl, cg_damping, cg_iter,
            queue, i, process_agent_count)
        workers.append(mp.Process(target=conjugate_gradient, args=worker_args))

    for worker in workers:
        worker.start()

    stepdirs = [None] * len(states_list)
    for _ in range(len(states_list)):
        pid, stepdir = queue.get(timeout=10)
        stepdirs[pid] = stepdir.numpy()

    queue.close()
    queue.join_thread()

    return stepdirs