# original : https://github.com/Mee321/policy-distillation

# ---------------------------------------------
# Autor : Junyeob Baek, wnsdlqjtm@naver.com
# ---------------------------------------------

from utils2.replay_memory import Memory
from utils2.torch import *
import ray
from core.running_state import ZFilter
from classroom import sample_generator

def merge_log(log_list):
    log = dict()
    log['total_reward'] = sum([x['total_reward'] for x in log_list])
    log['num_episodes'] = sum([x['num_episodes'] for x in log_list])
    log['num_steps'] = sum([x['num_steps'] for x in log_list])
    log['avg_reward'] = log['total_reward'] / log['num_episodes']
    log['max_reward'] = max([x['max_reward'] for x in log_list])
    log['min_reward'] = min([x['min_reward'] for x in log_list])
    if 'total_c_reward' in log_list[0]:
        log['total_c_reward'] = sum([x['total_c_reward'] for x in log_list])
        log['avg_c_reward'] = log['total_c_reward'] / log['num_steps']
        log['max_c_reward'] = max([x['max_c_reward'] for x in log_list])
        log['min_c_reward'] = min([x['min_c_reward'] for x in log_list])

    return log


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
        print("collect_samples called!!")
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


        # to_device(self.device, self.policies)
        # log['action_mean'] = np.mean(np.vstack(batch.action), axis=0)
        # log['action_min'] = np.min(np.vstack(batch.action), axis=0)
        # log['action_max'] = np.max(np.vstack(batch.action), axis=0)
        print("collect_samples done")
        return worker_memories, worker_logs

    def get_expert_sample(self, batch_size):
        print("get_expert_sample called!!")
        memories, logs = self.collect_samples(batch_size)
        teacher_rewards = [log['avg_reward'] for log in logs if log is not None]
        teacher_average_reward = np.array(teacher_rewards).mean()
        # TODO better implementation of dataset and sampling
        # construct training dataset containing pairs {X:state, Y:output of teacher policy}
        dataset = []
        for memory, policy in zip(memories, self.policies):
            batch = memory.sample()
            states = torch.from_numpy(np.stack(batch.state)).to(torch.double).to('cpu')
            means = policy.mean_action(states).detach()
            stds = policy.get_std(states).detach()
            dataset += [(state, mean, std) for state, mean, std in zip(states, means, stds)]
        return dataset, teacher_average_reward
