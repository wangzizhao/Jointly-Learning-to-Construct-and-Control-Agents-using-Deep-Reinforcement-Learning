from baselines.common import Dataset, explained_variance, fmt_row, zipsame
from baselines import logger
import baselines.common.tf_util as U
import tensorflow as tf, numpy as np
import time
from baselines.common.mpi_adam import MpiAdam
from baselines.common.mpi_moments import mpi_moments
from mpi4py import MPI
from collections import deque
import pdb

def episode_generator(pi, env, gamma, stochastic):
	ob = env.reset()
	R = 0

	while True:
		ac, _ = pi.act(stochastic, ob)
		# Slight weirdness here because we need value function at time T
		# before returning segment [0, T-1] so we get the correct
		# terminal value
		ob, rew, new, _ = env.step(ac)
		R += rew

		if new:
			return R
			

def traj_segment_generator(pi, env, horizon, stochastic):
	t = 0
	ac = env.action_space.sample() # not used, just so we have the datatype
	new = True # marks if we're on first timestep of an episode
	ob = env.reset()

	cur_ep_ret = 0 # return in current episode
	cur_ep_len = 0 # len of current episode
	ep_rets = [] # returns of completed episodes in this segment
	ep_lens = [] # lengths of ...

	# Initialize history arrays
	obs = np.array([ob for _ in range(horizon)])
	rews = np.zeros(horizon, 'float32')
	vpreds = np.zeros(horizon, 'float32')
	news = np.zeros(horizon, 'int32')
	acs = np.array([ac for _ in range(horizon)])
	prevacs = acs.copy()

	while True:
		prevac = ac
		ac, vpred = pi.act(stochastic, ob)
		# Slight weirdness here because we need value function at time T
		# before returning segment [0, T-1] so we get the correct
		# terminal value
		if t > 0 and t % horizon == 0:
			yield {"ob" : obs, "rew" : rews, "vpred" : vpreds, "new" : news,
					"ac" : acs, "prevac" : prevacs, "nextvpred": vpred * (1 - new),
					"ep_rets" : ep_rets, "ep_lens" : ep_lens}
			# Be careful!!! if you change the downstream algorithm to aggregate
			# several of these batches, then be sure to do a deepcopy
			ep_rets = []
			ep_lens = []
		i = t % horizon
		obs[i] = ob
		vpreds[i] = vpred
		news[i] = new
		acs[i] = ac
		prevacs[i] = prevac

		ob, rew, new, _ = env.step(ac)
		rews[i] = rew

		cur_ep_ret += rew
		cur_ep_len += 1
		if new:
			ep_rets.append(cur_ep_ret)
			ep_lens.append(cur_ep_len)
			cur_ep_ret = 0
			cur_ep_len = 0
			ob = env.reset()
		t += 1

def add_vtarg_and_adv(seg, gamma, lam):
	"""
	Compute target value using TD(lambda) estimator, and advantage with GAE(lambda)
	"""
	new = np.append(seg["new"], 0) # last element is only used for last vtarg, but we already zeroed it if last new = 1
	vpred = np.append(seg["vpred"], seg["nextvpred"])
	T = len(seg["rew"])
	seg["adv"] = gaelam = np.empty(T, 'float32')
	rew = seg["rew"]
	lastgaelam = 0
	for t in reversed(range(T)):
		nonterminal = 1-new[t+1]
		delta = rew[t] + gamma * vpred[t+1] * nonterminal - vpred[t]
		gaelam[t] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
	seg["tdlamret"] = seg["adv"] + seg["vpred"]

def learn(
		# =========== modified part begins =========== #
		env_id, seed,
		robot,						# robot class with GMM params
		joint_optimization_iters,	# total number of joint optimization iterations
		design_iters,				# number of samples when updating physical design in each joint optimization iteration
		policy_iters,				# number of samples when updating robot policy in each joint optimization iteration
		# ============ modified part ends ============ #

		policy_func, *,
		timesteps_per_actorbatch, # timesteps per actor per update
		clip_param, entcoeff, # clipping parameter epsilon, entropy coeff
		optim_epochs, optim_stepsize, optim_batchsize,# optimization hypers
		gamma, lam, # advantage estimation
		max_timesteps=0, max_episodes=0, max_iters=0, max_seconds=0,  # time constraint
		callback=None, # you can do anything in the callback, since it takes locals(), globals()
		adam_epsilon=1e-5,
		schedule='constant' # annealing for stepsize parameters (epsilon and adam)
		):

	# ================================== modification 1 ================================== #
	"""
	input:  replace "env" (env class) with "env_id" (string)
			add "seed" (int)
		reason: to enable env.make() during training
		modification detail: add following lines into learn()
			env = gym.make(env_id)
			env = bench.Monitor(env, logger.get_dir())
			env.seed(seed)
			env.close() # added at the end of learn()
	"""
	import roboschool, gym
	from baselines import bench
	env = gym.make(env_id)
	env = bench.Monitor(env, logger.get_dir())
	env.seed(seed)
	# ================================== modification 1 ================================== #


	# Setup losses and stuff
	# ----------------------------------------
	ob_space = env.observation_space
	ac_space = env.action_space

	# policy_func is the initialization of NN
	# NN structure:
	#   state -> (num_hid_layers) fully-connected layers with (hid_size) units -> (action, predicted value)
	#       num_hid_layers, hid_size: set in the file calls "learn"
	pi = policy_func("pi", ob_space, ac_space) # Construct network for new policy
	oldpi = policy_func("oldpi", ob_space, ac_space) # Network for old policy

	atarg = tf.placeholder(dtype=tf.float32, shape=[None]) # Target advantage function (if applicable)
	ret = tf.placeholder(dtype=tf.float32, shape=[None]) # Empirical return

	lrmult = tf.placeholder(name='lrmult', dtype=tf.float32, shape=[]) # learning rate multiplier, updated with schedule
	clip_param = clip_param * lrmult # Annealed cliping parameter epislon

	# placeholder for "ob"
	# created in mlppolicy.py
	ob = U.get_placeholder_cached(name="ob")
	# placeholder for "ac"
	# in common/distribution.py
	ac = pi.pdtype.sample_placeholder([None])

	# KL divergence and Entropy, defined in common/distribution.py
	kloldnew = oldpi.pd.kl(pi.pd)
	ent = pi.pd.entropy()
	meankl = U.mean(kloldnew)
	meanent = U.mean(ent)

	# pol_entpen: Entropy Bounus encourages exploration
	# entcoeff: entropy coefficient, defined in PPO page 5, Equ. (9)
	pol_entpen = (-entcoeff) * meanent

	# probability ration, defined in PPO page 3
	ratio = tf.exp(pi.pd.logp(ac) - oldpi.pd.logp(ac)) # pnew / pold

	# Surrogate Goal
	# defined in PPO page 3, Equ (7)
	surr1 = ratio * atarg # surrogate from conservative policy iteration
	surr2 = U.clip(ratio, 1.0 - clip_param, 1.0 + clip_param) * atarg #
	pol_surr = - U.mean(tf.minimum(surr1, surr2)) # PPO's pessimistic surrogate (L^CLIP)

	# Value Function Loss: square error loss for ||v_pred - v_target||
	vf_loss = U.mean(tf.square(pi.vpred - ret))

	# Total_loss = L^CLIP - Value Function Loss + Entropy Bounus
	# defined in PPO page 5, Equ. (9)
	total_loss = pol_surr + pol_entpen + vf_loss
	losses = [pol_surr, pol_entpen, vf_loss, meankl, meanent]
	loss_names = ["pol_surr", "pol_entpen", "vf_loss", "kl", "ent"]

	var_list = pi.get_trainable_variables()
	lossandgrad = U.function([ob, ac, atarg, ret, lrmult], losses + [U.flatgrad(total_loss, var_list)])
	# adam optimizer?
	adam = MpiAdam(var_list, epsilon=adam_epsilon)

	# oldpi = pi
	assign_old_eq_new = U.function([],[], updates=[tf.assign(oldv, newv)
		for (oldv, newv) in zipsame(oldpi.get_variables(), pi.get_variables())])

	# Why we need this line?
	compute_losses = U.function([ob, ac, atarg, ret, lrmult], losses)

	U.initialize()
	adam.sync()




	# ================================== modification 2 ================================== #
	for joint_optimization_iter in range(joint_optimization_iters):
		U.save_state('/home/zzwang/Desktop/EECS_498_008_Project/models/model_j{}_p{}_d{}_i{}.ckpt'.format(joint_optimization_iters, policy_iters, design_iters,joint_optimization_iter))
		logger.log("joint optimization progree: {}/{}".format(joint_optimization_iter, joint_optimization_iters))
		# ================================== update physical design ================================== #
		Rewards_plus = np.zeros(design_iters)
		Rewards_minum = np.zeros(design_iters)
		params = robot.sample(design_iters, to_update=True)
		for i, param in enumerate(params):
			robot.modify_file(param)
			env = gym.make(env_id)
			# myenv = env.env
			
			# pdb.set_trace()
			env = bench.Monitor(env, logger.get_dir())
			R = episode_generator(pi, env, gamma, stochastic=True)
			logger.log("\t update physical design: %d/%d, rew: %f"%(i, 2*design_iters, R))
			if i%2 == 0:
				Rewards_plus[int(i/2)] = R
			else:
				Rewards_minum[int(i/2)] = R
		logger.log("prev_mu: ", robot.params_mu)
		logger.log("prev_sig: ", robot.params_sig)
		robot.update(Rewards_plus, Rewards_minum)
		logger.log("mu: ", robot.params_mu)
		logger.log("sig: ", robot.params_sig)
		# ================================== update policy ================================== #
		# params = robot.sample(design_iters)
		params = [robot.params_mu]
		for param in params:
			# reinitialize env
			robot.modify_file(param)
			env = gym.make(env_id)
			env = bench.Monitor(env, logger.get_dir())
	# ================================== modification 2 ================================== #

			# Prepare for rollouts
			# ----------------------------------------
			seg_gen = traj_segment_generator(pi, env, timesteps_per_actorbatch, stochastic=True)

			episodes_so_far = 0
			timesteps_so_far = 0
			iters_so_far = 0
			tstart = time.time()
			lenbuffer = deque(maxlen=100) # rolling buffer for episode lengths
			rewbuffer = deque(maxlen=100) # rolling buffer for episode rewards

			assert sum([max_iters>0, max_timesteps>0, max_episodes>0, max_seconds>0])==1, "Only one time constraint permitted"

			while True:
				if callback: callback(locals(), globals())
				if max_timesteps and timesteps_so_far >= max_timesteps:
					break
				elif max_episodes and episodes_so_far >= max_episodes:
					break
				elif max_iters and iters_so_far >= max_iters:
					break
				elif max_seconds and time.time() - tstart >= max_seconds:
					break

				# annealing for stepsize parameters (epsilon and adam)
				if schedule == 'constant':
					cur_lrmult = 1.0
				elif schedule == 'linear':
					cur_lrmult =  max(1.0 - float(timesteps_so_far) / max_timesteps, 0)
				else:
					raise NotImplementedError

				logger.log("********** Iteration %i ************"%iters_so_far)

				seg = seg_gen.__next__()
				add_vtarg_and_adv(seg, gamma, lam)

				# ob, ac, atarg, ret, td1ret = map(np.concatenate, (obs, acs, atargs, rets, td1rets))
				ob, ac, atarg, tdlamret = seg["ob"], seg["ac"], seg["adv"], seg["tdlamret"]
				vpredbefore = seg["vpred"] # predicted value function before udpate
				atarg = (atarg - atarg.mean()) / atarg.std() # standardized advantage function estimate
				d = Dataset(dict(ob=ob, ac=ac, atarg=atarg, vtarg=tdlamret), shuffle=not pi.recurrent)
				optim_batchsize = optim_batchsize or ob.shape[0]

				if hasattr(pi, "ob_rms"): pi.ob_rms.update(ob) # update running mean/std for policy
				
				# oldpi = pi
				# set old parameter values to new parameter values
				assign_old_eq_new()
				logger.log("Optimizing...")
				logger.log(fmt_row(13, loss_names))
				# Here we do a bunch of optimization epochs over the data
				for _ in range(optim_epochs):
					losses = [] # list of tuples, each of which gives the loss for a minibatch
					for batch in d.iterate_once(optim_batchsize):
						*newlosses, g = lossandgrad(batch["ob"], batch["ac"], batch["atarg"], batch["vtarg"], cur_lrmult)
						adam.update(g, optim_stepsize * cur_lrmult) 
						losses.append(newlosses)
					logger.log(fmt_row(13, np.mean(losses, axis=0)))

				logger.log("Evaluating losses...")
				losses = []
				for batch in d.iterate_once(optim_batchsize):
					newlosses = compute_losses(batch["ob"], batch["ac"], batch["atarg"], batch["vtarg"], cur_lrmult)
					losses.append(newlosses)            
				meanlosses,_,_ = mpi_moments(losses, axis=0)
				logger.log(fmt_row(13, meanlosses))
				for (lossval, name) in zipsame(meanlosses, loss_names):
					logger.record_tabular("loss_"+name, lossval)
				logger.record_tabular("ev_tdlam_before", explained_variance(vpredbefore, tdlamret))
				lrlocal = (seg["ep_lens"], seg["ep_rets"]) # local values
				listoflrpairs = MPI.COMM_WORLD.allgather(lrlocal) # list of tuples
				lens, rews = map(flatten_lists, zip(*listoflrpairs))
				lenbuffer.extend(lens)
				rewbuffer.extend(rews)
				logger.record_tabular("EpLenMean", np.mean(lenbuffer))
				logger.record_tabular("EpRewMean", np.mean(rewbuffer))
				logger.record_tabular("EpThisIter", len(lens))
				episodes_so_far += len(lens)
				timesteps_so_far += sum(lens)
				iters_so_far += 1
				logger.record_tabular("EpisodesSoFar", episodes_so_far)
				logger.record_tabular("TimestepsSoFar", timesteps_so_far)
				logger.record_tabular("TimeElapsed", time.time() - tstart)
				if MPI.COMM_WORLD.Get_rank()==0:
					logger.dump_tabular()

	# ================================== modification 1 ================================== #
	env.close()
	# ================================== modification 1 ================================== #


def flatten_lists(listoflists):
	return [el for list_ in listoflists for el in list_]
