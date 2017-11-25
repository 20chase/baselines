#!/usr/bin/env python
import argparse
import os, sys
import numpy as np
from baselines import bench, logger

def train(env_id, num_timesteps, seed):
    from baselines.common import set_global_seeds
    from baselines.common.vec_env.vec_normalize import VecNormalize
    from baselines.ppo2 import ppo2
    from baselines.ppo2.policies import MlpPolicy
    import gym
    import roboschool
    import tensorflow as tf
    from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
    ncpu = 1
    config = tf.ConfigProto(allow_soft_placement=True,
                            intra_op_parallelism_threads=ncpu,
                            inter_op_parallelism_threads=ncpu)
    tf.Session(config=config).__enter__()
    def make_env():
        env = gym.make(env_id)
        env = bench.Monitor(env, logger.get_dir())
        return env
    env = DummyVecEnv([make_env])
    env = VecNormalize(env)

    set_global_seeds(seed)
    policy = MlpPolicy
    ppo2.learn(policy=policy, env=env, nsteps=2048, nminibatches=32,
        lam=0.95, gamma=0.99, noptepochs=10, log_interval=1,
        ent_coef=0.0,
        lr=3e-4,
        cliprange=0.2,
        total_timesteps=num_timesteps)

def test(env_id, num_timesteps, seed, curr_path):
    from baselines.common import set_global_seeds
    from baselines.common.vec_env.vec_normalize import VecNormalizeTest
    from baselines.ppo2 import ppo2
    from baselines.ppo2.policies import MlpPolicy
    import gym
    import roboschool
    import tensorflow as tf
    from baselines.common.vec_env.dummy_vec_env import DummyVecTestEnv

    ncpu = 1
    config = tf.ConfigProto(allow_soft_placement=True,
                            intra_op_parallelism_threads=ncpu,
                            inter_op_parallelism_threads=ncpu)
    tf.Session(config=config).__enter__()
    def make_env():
        env = gym.make(env_id)
        env = bench.Monitor(env, logger.get_dir())
        return env
    env = DummyVecTestEnv([make_env])
    running_mean = np.load('{}/log/mean.npy'.format(curr_path))
    running_var = np.load('{}/log/var.npy'.format(curr_path))
    env = VecNormalizeTest(env, running_mean, running_var)

    set_global_seeds(seed)
    policy = MlpPolicy

    ppo2.test(policy=policy, env=env, nsteps=2048, nminibatches=32, 
        load_path='{}/log/checkpoints/{}'.format(curr_path, '00450'))


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', help='environment ID', default='RoboschoolAnt-v1')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--num-timesteps', type=int, default=int(1e6))
    parser.add_argument('--train', type=bool, default=True)
    args = parser.parse_args()
    curr_path = sys.path[0]
    logger.configure(dir='{}/log'.format(curr_path))
    if args.train:
        train(args.env, num_timesteps=args.num_timesteps, seed=args.seed)
    else:
        test(args.env, num_timesteps=args.num_timesteps, seed=args.seed,
            curr_path=curr_path)


if __name__ == '__main__':
    main()

