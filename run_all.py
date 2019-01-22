import os
iterations = 2000000

# for length in [0.2, 0.3, 0.4, 0.5, 0.6]:
# 	os.system('python3 run_ant.py --length {} --num-timesteps {}'.format(length, iterations))

for d_iters in [30]:
	for p_iters in [1]:
		for j_iters in [300]:
			os.system('python run_roboschool.py --design-iters {} --policy-iters {} --joint-iters {}'.format(d_iters, p_iters, j_iters))

# for length in [0.3, 0.4, 0.5, 0.6, 0.7]:
# 	os.system('python3 run_walker2d.py --length {} --num-timesteps {}'.format(length, iterations))