# original : https://github.com/Mee321/policy-distillation

# ---------------------------------------------
# Autor : Junyeob Baek, wnsdlqjtm@naver.com
# ---------------------------------------------

from utils2.replay_memory import Memory
from utils2.torch import *
import ray
from core.running_state import ZFilter

@ray.remote
def collect_samples(pid, env, policy, custom_reward,
                    mean_action, render, running_state, min_batch_size):
    torch.randn(pid)
    log = dict()
    memory = Memory()
    num_steps = 0
    total_reward = 0
    min_reward = 1e6
    max_reward = -1e6
    total_c_reward = 0
    min_c_reward = 1e6
    max_c_reward = -1e6
    num_episodes = 0
    #running_state = ZFilter((env.observation_space.shape[0],), clip=5)
    running_state = None
    while num_steps < min_batch_size:
        state = env.reset()
        if running_state is not None:
            state = running_state(state)
        reward_episode = 0

        for t in range(10000):
            state_var = tensor(state).unsqueeze(0)
            with torch.no_grad():
                if mean_action:
                    action = policy(state_var)[0][0].numpy()
                else:
                    action = policy.select_action(state_var)[0].numpy()
            action = int(action) if policy.is_disc_action else action.astype(np.float64)
            next_state, reward, done, _ = env.step(action)
            reward_episode += reward
            if running_state is not None:
                next_state = running_state(next_state)

            if custom_reward is not None:
                reward = custom_reward(state, action)
                total_c_reward += reward
                min_c_reward = min(min_c_reward, reward)
                max_c_reward = max(max_c_reward, reward)

            mask = 0 if done else 1

            memory.push(state, action, mask, next_state, reward)

            if render:
                env.render()
            if done:
                break

            state = next_state

        # log stats
        num_steps += (t + 1)
        num_episodes += 1
        total_reward += reward_episode
        min_reward = min(min_reward, reward_episode)
        max_reward = max(max_reward, reward_episode)

    log['num_steps'] = num_steps
    log['num_episodes'] = num_episodes
    log['total_reward'] = total_reward
    log['avg_reward'] = total_reward / num_episodes
    log['max_reward'] = max_reward
    log['min_reward'] = min_reward
    if custom_reward is not None:
        log['total_c_reward'] = total_c_reward
        log['avg_c_reward'] = total_c_reward / num_steps
        log['max_c_reward'] = max_c_reward
        log['min_c_reward'] = min_c_reward

    return (pid, memory, log)


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
        for policy in self.policies:
            to_device(torch.device('cpu'), policy)
        result_ids = []
        for i in range(self.num_teachers):
            result_ids.append(collect_samples.remote(i, self.envs[i], self.policies[i], self.custom_reward, self.mean_action,
                           False, self.running_state, min_batch_size))
        worker_logs = [None] * self.num_agents
        worker_memories = [None] * self.num_agents
        # print(len(result_ids))
        for result_id in result_ids:
            pid, worker_memory, worker_log = ray.get(result_id)
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
