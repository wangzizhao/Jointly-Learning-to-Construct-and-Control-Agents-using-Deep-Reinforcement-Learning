#!/usr/bin/env python
from baselines.common import set_global_seeds, tf_util as U
from baselines import bench
import gym, logging
from baselines import logger
import roboschool
from Gaussian_Mixture import GMM

def train(args):

    env_id = args.env
    num_timesteps = args.num_timesteps
    seed = args.seed

    from baselines.ppo1 import mlp_policy
    import pposgd_simple_modified_final
    U.make_session(num_cpu=1).__enter__()

    # set random seed for tf, numpy.random, random
    # in common/misc_util.py
    set_global_seeds(seed)

    def policy_fn(name, ob_space, ac_space):
    	# mlp: Multi-Layer Perceptron
    	# state -> (num_hid_layers) fully-connected layers with (hid_size) units -> (action, predicted value)
        return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
            hid_size=64, num_hid_layers=2)

    # ================================== modification 1 ================================== #
    """
    ppo_learn input: replace "env" (env class) with "env_id" (string)
    				 add input "seed" (int)
    	reason: to enable env.make() during training
    	modification detail: move following lines into learn()
    		env = gym.make(env_id)
    		env = bench.Monitor(env, logger.get_dir())
    		env.seed(seed)
    		env.close()
	"""

    # ====================================== hyperparameter begins ====================================== #
    joint_optimization_iters = args.joint_iters
    design_iters = args.design_iters                    # number of robots sampled when updating physical design
    policy_iters = args.policy_iters                    # number of robots sampled when updating policy
    policy_episodes = 1                 # for each robot, number of epsiodes conducted to update policy
    policy_timesteps = 1e5
    design_learning_rate = 1e-4
    # ======================================= hyperparameter ends ======================================= #
    if 'Ant' in env_id:
        robot_name = 'ant'
    elif 'Hopper' in env_id:
        robot_name = 'hopper'
    elif 'Walker' in env_id:
        robot_name = 'Walker2d'
    else:
        print ('!'*50)
        print ('Unknown Environment')
        print ('!'*50)
        exit(1)


    robot = GMM(robot_name = robot_name, m = design_iters, learning_rate = design_learning_rate)

    # ================================== modification 1 ================================== #
    gym.logger.setLevel(logging.WARN)
    pposgd_simple_modified_final.learn(
            # =========== modified part begins =========== #
            env_id, seed, 
            robot,                      # robot class with GMM params
            joint_optimization_iters,   # total number of joint optimization iterations
            design_iters,               # number of samples when updating physical design in each joint optimization iteration
            policy_iters,
            # ============ modified part ends ============ #
            policy_fn,
            max_timesteps=policy_timesteps,
            timesteps_per_actorbatch=2048,
            clip_param=0.2, entcoeff=0.0,
            optim_epochs=10, optim_stepsize=3e-4, optim_batchsize=64,
            gamma=0.99, lam=0.95, schedule='linear',
        )

def main():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', help='environment ID', default='RoboschoolAnt-v1')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--num-timesteps', type=int, default=int(1e6))
    parser.add_argument('--design-iters', type=int, default=int(10))   
    parser.add_argument('--policy-iters', type=int, default=int(10))
    parser.add_argument('--joint-iters', type=int, default=int(100))


    args = parser.parse_args()
    logger_dir = '/home/zzwang/Desktop/EECS_498_008_Project/logger/ant_joint_opt_j'+str(args.joint_iters)+'_d'+str(args.design_iters)+'_p'+str(args.policy_iters)
    logger.configure(dir = logger_dir)
    train(args)


if __name__ == '__main__':
    main()
