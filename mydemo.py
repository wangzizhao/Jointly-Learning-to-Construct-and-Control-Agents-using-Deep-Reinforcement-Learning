import os.path, gym
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import roboschool
import pdb

from baselines.common import set_global_seeds, tf_util as U
from baselines import bench
import gym, logging
from Gaussian_Mixture import GMM
from baselines.ppo1 import mlp_policy
import pposgd_simple_modified_final

# ====================== modified ======================= #

def get_session():
    """Returns recently made Tensorflow session"""
    return tf.get_default_session()

def load_state(fname):
    saver = tf.train.Saver()
    saver.restore(get_session(), fname)
    # saver.restore(sess, fname)

def save_state(fname):
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    saver = tf.train.Saver()
    saver.save(get_session(), fname)

def policy_fn(name, ob_space, ac_space):
    # mlp: Multi-Layer Perceptron
    # state -> (num_hid_layers) fully-connected layers with (hid_size) units -> (action, predicted value)
    return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
        hid_size=64, num_hid_layers=2)

# ====================== modified ======================= #




def demo_run():


    config = tf.ConfigProto(
        inter_op_parallelism_threads=1,
        intra_op_parallelism_threads=1,
        device_count = { "GPU": 0 } )
    sess = tf.InteractiveSession(config=config)

    env = gym.make("RoboschoolAnt-v1")


    pi = policy_fn("pi", env.observation_space, env.action_space) # Construct network for new policy
    load_state('/home/yetong/Desktop/Project/models/model1.ckpt')






    # pi = ZooPolicyTensorflow("mymodel1", env.observation_space, env.action_space)

    while 1:
        frame = 0
        score = 0
        restart_delay = 0
        obs = env.reset()

        while 1:
            a,_ = pi.act(True, obs)

            # print (a)
            # pdb.set_trace()

            obs, r, done, _ = env.step(a)


            score += r
            frame += 1
            still_open = env.render("human")
            if still_open==False:
                return
            if not done: continue
            if restart_delay==0:
                print("score=%0.2f in %i frames" % (score, frame))
                if still_open!=True:      # not True in multiplayer or non-Roboschool environment
                    break
                restart_delay = 60*2  # 2 sec at 60 fps
            restart_delay -= 1
            if restart_delay==0: break

if __name__=="__main__":
    demo_run()
