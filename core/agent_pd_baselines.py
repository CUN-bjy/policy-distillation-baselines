# original : https://github.com/Mee321/policy-distillation

# ---------------------------------------------
# Autor : Junyeob Baek, wnsdlqjtm@naver.com
# ---------------------------------------------

from utils2.replay_memory import Memory
from utils2.torch import *
import ray
from core.running_state import ZFilter
from classroom import sample_generator

class AgentCollection:

    def __init__(self, envs, policies, device, custom_reward=None,
                 mean_action=False, render=False, running_state=None, num_agents=1, num_parallel_workers=1):
        self.envs = envs
        self.policies = policies
        self.device = device
        self.custom_reward = custom_reward
        self.mean_action = mean_action
        #self.running_state = running_state
        self.running_state = None
        self.render = render
        self.num_parallel_workers = num_parallel_workers
        self.num_agents = num_agents
        self.num_teachers = len(policies)

    def collect_samples(self, min_batch_size):
        # print("collect_samples called!!")
        results = []
        for i in range(self.num_teachers):
            results.append(sample_generator(self.envs[i], self.policies[i], False, min_batch_size, i))
        worker_logs = [None] * self.num_agents
        worker_memories = [None] * self.num_agents
        # print(len(result_ids))
        for result in results:
            pid, worker_memory, worker_log = result
            worker_memories[pid] = worker_memory
            worker_logs[pid] = worker_log

        # print("collect_samples done")
        return worker_memories, worker_logs

    def get_expert_sample(self, batch_size, deterministic=True):
        # print("get_expert_sample called!!")
        memories, logs = self.collect_samples(batch_size)
        teacher_rewards = [log['avg_reward'] for log in logs if log is not None]
        teacher_average_reward = np.array(teacher_rewards).mean()
        # TODO better implementation of dataset and sampling
        # construct training dataset containing pairs {X:state, Y:output of teacher policy}
        dataset = []
        for memory, policy in zip(memories, self.policies):
            batch = memory.sample()
            batched_state = np.array(batch.state).reshape(-1, policy.env.observation_space.shape[0])
            states = torch.from_numpy(batched_state).to(torch.double).to('cpu')
            act_dist = torch.from_numpy(policy.predict(states, deterministic=deterministic)[0])
            dataset += [(state, act_dist) for state, act_dist in zip(states, act_dist)]
        return dataset, teacher_average_reward
