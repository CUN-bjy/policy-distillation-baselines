# original : https://github.com/Mee321/policy-distillation

# ---------------------------------------------
# Autor : Junyeob Baek, wnsdlqjtm@naver.com
# ---------------------------------------------

import gym
from utils.models import *
from torch.optim import Adam, SGD
import torch
from torch.autograd import Variable
import random
from utils.math import get_wasserstein, get_kl
from utils.agent_pd_baselines import AgentCollection
import numpy as np
from copy import deepcopy

from classroom import load_env_and_model

class Student(object):
    def __init__(self, env, args, optimizer=None):

        env_id = 'AntBulletEnv-v0'
        algo = 'td3'
        folder = "rl-trained-agents"
        self.env, _ = load_env_and_model(env_id, algo, folder)

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
        # print(batch[0])
        states = torch.stack([x[0] for x in batch])
        means_teacher = torch.stack([x[1] for x in batch])

        fake_std = torch.from_numpy(np.array([1e-6]*len(means_teacher[0]))) # for deterministic
        stds_teacher = torch.stack([fake_std for x in batch])

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
        memories, logs = self.agents.collect_samples(self.testing_batch_size, exercise=True)
        rewards = [log['avg_reward'] for log in logs]
        average_reward = np.array(rewards).mean()
        return average_reward


class Teacher(object):
    def __init__(self, envs, policies, args):
        self.envs = envs
        self.policies = policies
        self.expert_batch_size = args.sample_batch_size
        self.agents = AgentCollection(self.envs, self.policies, 'cpu', running_state=None, render=args.render,
                                      num_agents=args.agent_count, num_parallel_workers=args.num_workers)

    def get_expert_sample(self):
        return self.agents.get_expert_sample(self.expert_batch_size)
