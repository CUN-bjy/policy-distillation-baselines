# original : https://github.com/Mee321/policy-distillation

import gym
from core.models import *
from torch.optim import Adam, SGD
import torch
from torch.autograd import Variable
import random
from utils2.math import get_wasserstein, get_kl
from core.agent_pd_baselines import AgentCollection
import numpy as np
from utils.torch import *
from copy import deepcopy

class Student(object):
    def __init__(self, args, optimizer=None):
        self.env = gym.make(args.env_name)
        num_inputs = self.env.observation_space.shape[0]
        num_actions = self.env.action_space.shape[0]
        self.training_batch_size = args.student_batch_size
        self.testing_batch_size = args.testing_batch_size
        self.loss_metric = args.loss_metric
        self.policy = Policy(num_inputs, num_actions, hidden_sizes=(args.hidden_size,) * args.num_layers)
        self.agents = AgentCollection([self.env], [self.policy], 'cpu', running_state=None, render=args.render,
                                        num_agents=1, num_parallel_workers=1)
        if not optimizer:
            self.optimizer = SGD(self.policy.parameters(), lr=args.lr)

    def train(self, expert_data):
        batch = random.sample(expert_data, self.training_batch_size)
        print(batch[0])
        states = torch.stack([x[0] for x in batch])
        means_teacher = torch.stack([x[1] for x in batch])
        stds_teacher = torch.stack(torch.from_numpy(np.array([[0]*len(x[1]) for x in batch])))
        means_student = self.policy.mean_action(states)
        stds_student = self.policy.get_std(states)
        if self.loss_metric == 'kl':
            loss = get_kl([means_teacher, stds_teacher], [means_student, stds_student])
        elif self.loss_metric == 'wasserstein':
            loss = get_wasserstein([means_teacher, stds_teacher], [means_student, stds_student])
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss

    def test(self):
        memories, logs = self.agents.collect_samples(self.testing_batch_size)
        rewards = [log['avg_reward'] for log in logs]
        average_reward = np.array(rewards).mean()
        return average_reward

    def npg_train(self, expert_data):
        batch = random.sample(expert_data, self.training_batch_size)
        states = torch.stack([x[0] for x in batch])
        means_teacher = torch.stack([x[1] for x in batch])
        stds_teacher = torch.stack([x[2] for x in batch])
        means_student = self.policy.mean_action(states)
        stds_student = self.policy.get_std(states)

        def _fvp(states, damping=1e-2):
            def __fvp(vector, damping=damping):
                pi = self.policy(Variable(states))
                pi_detach = detach_distribution(pi)
                kl = torch.mean(kl_divergence(pi_detach, pi))
                grads = torch.autograd.grad(kl, self.policy.parameters(), create_graph=True)
                flat_grad_kl = torch.cat([grad.view(-1) for grad in grads])

                kl_v = (flat_grad_kl * vector).sum()
                grads = torch.autograd.grad(kl_v, self.policy.parameters())
                flat_grad_grad_kl = torch.cat([grad.view(-1) for grad in grads])

                return flat_grad_grad_kl + vector * damping

            return __fvp

        if self.loss_metric == 'kl':
            loss = get_kl([means_teacher, stds_teacher], [means_student, stds_student])
        elif self.loss_metric == 'wasserstein':
            loss = get_wasserstein([means_teacher, stds_teacher], [means_student, stds_student])
        grad = torch.autograd.grad(loss, self.policy.parameters(), create_graph=True)
        grad = flat(grad).detach()

        fvp = _fvp(states, damping=1e-2)
        stepdir = cg(fvp, grad, 10)
        shs = 0.5 * (stepdir * fvp(stepdir)).sum(0, keepdim=True)
        lm = torch.sqrt(shs / 1e-2)
        fullstep = stepdir / lm[0]
        prev_params = get_flat_params_from(self.policy)
        updated_params = prev_params - fullstep
        set_flat_params_to(self.policy, updated_params)

        return loss

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
