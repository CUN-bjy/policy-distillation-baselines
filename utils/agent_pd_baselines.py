# original : https://github.com/Mee321/policy-distillation

# ---------------------------------------------
# Autor : Junyeob Baek, wnsdlqjtm@naver.com
# ---------------------------------------------

from utils.replay_memory import Memory
from utils.torch import *
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

    def collect_samples(self, min_batch_size, exercise=False):
        # print("collect_samples called!!")
        results = []
        for i in range(self.num_teachers):
            if not exercise:
                results.append(sample_generator(self.envs[i], self.policies[i], False, min_batch_size, i))
            else:
                results.append(self.exercise(self.envs[i], self.policies[i], False, min_batch_size, i))
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
            states = torch.from_numpy(batched_state).to(torch.float).to('cpu')
            act_dist = torch.from_numpy(policy.predict(states, deterministic=deterministic)[0])
            dataset += [(state, act_dist) for state, act_dist in zip(states, act_dist)]
        return dataset, teacher_average_reward

    def exercise(self, env, policy, render=True, min_batch_size=10000, pid=0):
        torch.randn(pid)
        log = dict()
        memory = Memory()
        num_steps = 0
        total_reward = 0
        min_reward = 1e6
        max_reward = -1e6
        num_episodes = 0

        while num_steps < min_batch_size:
            state = env.reset()
            reward_episode = 0

            for t in range(1000):
                state_var = tensor(state).unsqueeze(0)
                with torch.no_grad():
                    action = policy.mean_action(state_var.to(torch.float))[0].numpy()
                next_state, reward, done, _ = env.step(action)
                reward_episode += reward

                mask = 0 if done else 1

                memory.push(state, action, mask, next_state, reward)

                if render:
                    env.render()
                if done:
                    break

                state = next_state

            # log states
            num_steps += (t + 1)
            num_episodes += 1
            total_reward += reward_episode
            min_reward = min(min_reward, reward_episode)
            max_reward = max(max_reward, reward_episode)
            print("reward_episode: %f"%reward_episode)
            print("num_steps: %d"%num_steps)


        log['num_steps'] = num_steps
        log['num_episodes'] = num_episodes
        log['total_reward'] = total_reward
        log['avg_reward'] = total_reward / num_episodes
        log['max_reward'] = max_reward
        log['min_reward'] = min_reward

        return (pid, memory, log)