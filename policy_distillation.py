# original : https://github.com/Mee321/policy-distillation

# ---------------------------------------------
# Autor : Junyeob Baek, wnsdlqjtm@naver.com
# ---------------------------------------------


# basic utils
from itertools import count
from time import time, strftime, localtime
import os, numpy as np
import scipy.optimize

# deeplearning utils
from tensorboardX import SummaryWriter
from core.models import *
from core.agent_ray_pd import AgentCollection
from utils.utils import *

# LDE utils
import gym
from classroom import load_env_and_model, sample_generator

# teacher policy & student policy
from student import Student
from teacher import Teacher


torch.utils.backcompat.broadcast_warning.enabled = True
torch.utils.backcompat.keepdim_warning.enabled = True

'''
1. train single or multiple teacher policies
2. collect samples from teacher policy
3. use KL or W2 distance as metric to train student policy
4. test student policy
'''

def main(args):
    # ray.init(num_cpus=args.num_workers, num_gpus=1)

    # policy and envs for sampling
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    exp_date = strftime('%Y.%m.%d', localtime(time()))
    writer = SummaryWriter(log_dir='./exp_data/{}/{}_{}'.format(exp_date, args.env_name, time()))


    ###########################################
    # load teacher models
    ##########################################
    envs = []
    teacher_policies = []
    for i in range(args.num_teachers):
        env_id = 'AntBulletEnv-v0'
        algo = 'td3'
        folder = "rl-trained-agents"

        env, model = load_env_and_model(env_id, algo, folder)
        envs.append(env)
        teacher_policies.append(model)
    ##########################################

    teachers = Teacher(envs, teacher_policies, args)
    student = Student(envs[0],args)
    print('Training student policy...')
    time_beigin = time()

    ################################
    # train student policy
    ################################
    for iter in count(1):
        if iter % args.sample_interval == 1:
            expert_data, expert_reward = teachers.get_expert_sample()
        if args.algo == 'npg':
            loss = student.npg_train(expert_data)
        else:
            loss = student.train(expert_data)
        writer.add_scalar('{} loss'.format(args.loss_metric), loss.data, iter)
        print('Itr {} {} loss: {:.2f}'.format(iter, args.loss_metric, loss.data))
        if iter % args.test_interval == 0:
            average_reward = student.test()
            writer.add_scalar('Students_average_reward', average_reward, iter)
            writer.add_scalar('teacher_reward', expert_reward, iter)
            print("Students_average_reward: {:.3f} (teacher_reaward:{:3f})".format(average_reward, expert_reward))
        if iter > args.num_student_episodes:
            break
    time_train = time() - time_beigin
    print('Training student policy finished, using time {}'.format(time_train))
    # ray.shutdown()


if __name__ == '__main__':
    import argparse

    ##########################
    # Arguments Setting
    ##########################
    parser = argparse.ArgumentParser(description='Policy distillation')

    # Network, env, seed
    parser.add_argument('--hidden-size', type=int, default=64,
                        help='number of hidden units per layer')
    parser.add_argument('--num-layers', type=int, default=2,
                        help='number of hidden layers')
    parser.add_argument('--env-name', default="'AntBulletEnv-v0'", metavar='G',
                        help='name of the environment to run')
    parser.add_argument('--seed', type=int, default=1, metavar='N',
                        help='random seed (default: 1)')

    # For Teacher policy
    parser.add_argument('--agent-count', type=int, default=3, metavar='N',
                        help='number of agents (default: 100)')
    parser.add_argument('--num-teachers', type=int, default=1, metavar='N',
                        help='number of teacher policies (default: 1)')
    parser.add_argument('--sample-batch-size', type=int, default=10000, metavar='N',
                        help='expert batch size for each teacher (default: 10000)')
    parser.add_argument('--render', action='store_true',
                        help='render the environment')
    parser.add_argument('--num-workers', type=int, default=3,
                        help='number of workers for parallel computing')

    # For Student policy
    parser.add_argument('--lr', type=float, default=1e-3, metavar='G',
                        help='adam learnig rate (default: 1e-3)')
    parser.add_argument('--test-interval', type=int, default=10, metavar='N',
                        help='interval between training status logs (default: 10)')
    parser.add_argument('--student-batch-size', type=int, default=1000, metavar='N',
                        help='per-iteration batch size for student (default: 1000)')
    parser.add_argument('--sample-interval', type=int, default=10, metavar='N',
                        help='frequency to update expert data (default: 10)')
    parser.add_argument('--testing-batch-size', type=int, default=5000, metavar='N',
                        help='batch size for testing student policy (default: 10000)')
    parser.add_argument('--num-student-episodes', type=int, default=1000, metavar='N',
                        help='num of teacher training episodes (default: 1000)')
    parser.add_argument('--loss-metric', type=str, default='kl',
                        help='metric to build student objective')
    parser.add_argument('--algo', type=str, default='sgd',
                        help='update method')
    args = parser.parse_args()


    ########################
    # Distilling!
    ########################
    # try:
    main(args)
    # except Exception as e:
    #     ray.shutdown()
    #     print(e)
