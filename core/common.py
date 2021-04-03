import torch
from utils2.torch import to_device
import numpy as np
import multiprocessing as mp
import math


def estimate_advantages(memories, value_net, gamma, tau, device='cpu', dtype=torch.double, queue=None, pid=None, num_agent=None):
    advantages_list = []
    states_list =[]
    actions_list = []
    returns_list = []

    for memory, memory_index in zip(memories, range(len(memories))):
        batch = memory.sample()
        states = torch.from_numpy(np.stack(batch.state)).to(dtype).to(device)
        rewards = torch.from_numpy(np.stack(batch.reward)).to(dtype).to(device)
        actions = torch.from_numpy(np.stack(batch.action)).to(dtype).to(device)
        masks = torch.from_numpy(np.stack(batch.mask)).to(dtype).to(device)

        # with torch.no_grad():
        values = value_net(states).detach()
        rewards, masks, values = to_device(torch.device('cpu'), rewards, masks, values)
        tensor_type = type(rewards)
        deltas = tensor_type(rewards.size(0), 1)
        advantages = tensor_type(rewards.size(0), 1)
        prev_value = 0
        prev_advantage = 0
        for i in reversed(range(rewards.size(0))):
            deltas[i] = rewards[i] + gamma * prev_value * masks[i] - values[i]
            advantages[i] = deltas[i] + gamma * tau * prev_advantage * masks[i]

            prev_value = values[i, 0]
            prev_advantage = advantages[i, 0]

        returns = values + advantages
        advantages = (advantages - advantages.mean()) / advantages.std()
        advantages, returns, states, actions = to_device(device, advantages, returns, states, actions)


        states_list.append(states)
        actions_list.append(actions)
        advantages_list.append(advantages)
        returns_list.append(returns)

        if queue is not None:
            queue.put([pid*num_agent+memory_index, advantages, returns, states, actions])

    if queue is None:
        return advantages_list, returns_list, states_list, actions_list


def estimate_advantages_parallel(memories, value_net, gamma, tau, device='cpu', dtype=torch.float64, num_parallel_workers=mp.cpu_count()):
    workers = []
    queue = mp.Queue()
    num_agents = len(memories)
    process_agent_count = int(math.floor(num_agents / num_parallel_workers))
    for i in range(num_parallel_workers):
        worker_args = (memories[i*process_agent_count:(i+1)*process_agent_count], value_net, gamma, tau, device, dtype, queue, i, process_agent_count)
        workers.append(mp.Process(target=estimate_advantages, args=worker_args))

    for worker in workers:
        worker.start()

    advantages_list = [None]*len(memories)
    returns_list = [None]*len(memories)
    states_list = [None] * len(memories)
    actions_list = [None] * len(memories)
    for _ in range(len(memories)):
        pid, advantages, returns, states, actions = queue.get(timeout=10)
        # print("pid {}. done".format(pid))
        advantages_list[pid] = advantages
        returns_list[pid] = returns
        states_list[pid] = states
        actions_list[pid] = actions

    queue.close()
    queue.join_thread()
    return advantages_list, returns_list, states_list, actions_list

