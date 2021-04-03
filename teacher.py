from core.agent_ray_pd import AgentCollection


class Teacher(object):
    def __init__(self, envs, policies, args):
        self.envs = envs
        self.policies = policies
        self.expert_batch_size = args.sample_batch_size
        self.agents = AgentCollection(self.envs, self.policies, 'cpu', running_state=None, render=args.render,
                                      num_agents=args.agent_count, num_parallel_workers=args.num_workers)

    def get_expert_sample(self):
        return self.agents.get_expert_sample(self.expert_batch_size)
