import matplotlib.pyplot as plt
import os
import re, pdb


def get_rewards(file_path, total_iters=1000):

    with open(file_path, 'r') as f:
        lines=f.read().splitlines()
    line_iter = iter(lines)

    num_iters = 0
    my_iter = 0

    losses = []
    iters_list = []
    rewards = []
    prev_timesteps = 0
    try:
        line = next(line_iter)
        while num_iters < total_iters:
            while not line.startswith('Evaluating'):
                line = next(line_iter)
            line = next(line_iter)
            this_loss = re.findall(r"[-+]?\d*\.\d+|\d+",line)
            this_loss = [float(tmp) for tmp in this_loss]

            while not line.startswith('*******'):
                line = next(line_iter)
            num_iters = int(re.findall(r"[-+]?\d*\.\d+|\d+",line)[0])

            while not line.startswith('| EpRewMean'):
                line = next(line_iter)
            reward = float(line.rpartition('|')[0].rpartition('|')[2])

            while not line.startswith('| TimestepsSoFar'):
                line = next(line_iter)
            timesteps = float(line.rpartition('|')[0].rpartition('|')[2])

            last_timesteps = 0 if len(iters_list) == 0 else iters_list[-1]
            if timesteps>prev_timesteps:
                iters_list.append(timesteps-prev_timesteps+last_timesteps)
            else:
                iters_list.append(timesteps+last_timesteps)
            prev_timesteps = timesteps
            losses.append(this_loss)
            rewards.append(reward)

    except:
        pass

    return iters_list, rewards, losses


plt.figure('ant')
file_path = '/home/yetong/Desktop/Project/logger/ant_joint_opt_j300_d30_p1/log.txt'
iters_list, rewards, _ = get_rewards(file_path)
# iters_list = [i/1e8 for i in iters_list]
plt.plot(iters_list, rewards)


file_path = '/home/yetong/Desktop/Project/logger/ant_0.4/log.txt'
iters_list, rewards, _ = get_rewards(file_path)
# iters_list = [i/1e8 for i in iters_list]
plt.plot(iters_list, rewards)

plt.legend(['joint optimization', 'PPO baseline'], bbox_to_anchor=(0., 1.02, 1., .102))

plt.title('Ant')
plt.xlabel('number of timesteps')
plt.ylabel('episode mean reward')
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
print (iters_list)
print (rewards)
plt.show()
# plt.figure('ant')
# legends = []
# for length in [0.2, 0.3, 0.4, 0.5, 0.6]:
#   file_path = '/home/yetong/Desktop/Project/logger/ant_'+str(length)+'/log.txt'
#   iters_list, rewards, _ = get_rewards(file_path)
#   plt.plot(iters_list, rewards)
#   legends.append('leg '+str(length))
# plt.legend(legends)
# plt.title('Ant')
# plt.xlabel('number of episodes')
# plt.ylabel('episode mean reward')
# plt.savefig('ant.png')
# plt.close('ant')


# plt.figure('hopper')
# legends = []
# for length in [0.3, 0.4, 0.5, 0.6, 0.7]:
#   file_path = '/home/yetong/Desktop/Project/logger/hopper_'+str(length)+'/log.txt'
#   iters_list, rewards, _ = get_rewards(file_path)
#   plt.plot(iters_list, rewards)
#   legends.append('leg '+str(length))
# plt.legend(legends)
# plt.title('Hopper')
# plt.xlabel('number of episodes')
# plt.ylabel('episode mean reward')
# plt.savefig('hopper.png')
# plt.close('hopper')


# plt.figure('Walker2d')
# legends = []
# for length in [0.3, 0.4, 0.5, 0.6, 0.7]:
#   file_path = '/home/yetong/Desktop/Project/logger/walker2d_'+str(length)+'/log.txt'
#   iters_list, rewards, _ = get_rewards(file_path)
#   plt.plot(iters_list, rewards)
#   legends.append('leg '+str(length))
# plt.legend(legends)
# plt.title('Walker2d')
# plt.xlabel('number of episodes')
# plt.ylabel('episode mean reward')
# plt.savefig('Walker2d.png')
# plt.close('Walker2d')
# plt.plot(iters_list, pol_surr)
# plt.plot(iters_list, vf_loss)
# plt.plot(iters_list, kl)
# plt.plot(iters_list, ent)
# plt.legend(['pol_surr', 'vf_loss', 'kl', 'ent'])
# plt.show()
# print (iters_list)
# print (pol_surr)
# print (vf_loss)
# print (kl)
# print (ent)