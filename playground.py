# ---------------------------------------------
# Autor : Junyeob Baek, wnsdlqjtm@naver.com
# ---------------------------------------------

import argparse
from classroom import Alumni
from utils.agent_pd_baselines import load_env_and_model, sample_generator
from utils2 import ALGOS


if __name__ == '__main__':
	# arguments setting
	parser = argparse.ArgumentParser()
	parser.add_argument("-m", "--mode", help="playground mode", type=str, default='alumni', required=True, choices=list(['teacher', 'alumni']))
	parser.add_argument("--env", help="environment ID", type=str, default='AntBulletEnv-v0')
	parser.add_argument("--algo", help="RL Algorithm", default="td3", type=str, required=False, choices=list(ALGOS.keys()))
	parser.add_argument("-f", "--folder", type=str, default="rl-trained-agents",help='well trained teachers storage')
	parser.add_argument("-p", "--path-to-student", type=str, default="distilled-agents", help='well trained students sotrage')
	parser.add_argument('--render', type=bool, default=True, help='render the environment')
	parser.add_argument('--testing-batch-size', type=int, default=1000, metavar='N',
						help='batch size for testing student policy (default: 10000)')
	args = parser.parse_args()

	if args.mode is 'teacher':
		# play teacher    
		env, teacher = load_env_and_model(args.env, args.algo, args.folder)
		sample_generator(env, teacher, min_batch_size=args.testing_batch_size)
	else:
		# play alumni(trained-student)
		distilled_agent = Alumni(args)
		average_reward = distilled_agent.test()