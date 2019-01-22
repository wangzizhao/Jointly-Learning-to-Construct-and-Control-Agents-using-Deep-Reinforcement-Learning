import os

######################################## UPDATE LINE FOR ANT ########################################
def update_line_ant (lines, line_idx, len_part, rad_part, direction, mode=1): # mode = 1 for update link, mode = 2 for update axis
	line = lines[line_idx]
	if mode == 1:
		line_part1 = line.partition('fromto="')[0] + 'fromto="'
		line_part2 = '" name=' + line.partition('" name=')[2].partition('size="')[0] + 'size="'
		line_part3 = '" type' + line.partition('" type')[2]

		updated_part1 = "0.0 0.0 0.0 "+str(len_part*direction[0])+" "+str(len_part*direction[1])+" 0.0"
		updated_part2 = str(rad_part)
		line_updated = line_part1 + updated_part1 + line_part2 + updated_part2 + line_part3

	if mode == 2:
		line_part1 = line.partition('pos="')[0] + 'pos="'
		line_part2 = '"' + line.partition('pos="')[2].partition('"')[2]
		updated_part1 = str(len_part * direction[0]) + ' ' + str(len_part * direction[1]) +' 0'
		line_updated = line_part1 + updated_part1 + line_part2

	lines[line_idx] = line_updated
	# print (line_updated)

######################################## UPDATE LINE FOR HOPPER ########################################
def update_line_hopper (lines, line_idx, x1, z1, x2, z2, rad, mode=1): # mode = 1 for update link, mode = 2 for update axis
	line = lines[line_idx]
	if mode == 1:
		line_part1 = line.partition('fromto="')[0] + 'fromto="'
		line_part2 = '" name=' + line.partition('" name=')[2].partition('size="')[0] + 'size="'
		line_part3 = '" type' + line.partition('" type')[2]

		updated_part1 = str(x1) + ' 0 ' + str(z1) + ' ' + str(x2) + ' 0 '+str(z2)
		updated_part2 = str(rad)
		line_updated = line_part1 + updated_part1 + line_part2 + updated_part2 + line_part3

	if mode == 2:
		line_part1 = line.partition('pos="')[0] + 'pos="'
		line_part2 = '" range' + line.partition('" range')[2]
		updated_part1 = str(x2) + ' 0 ' + str(z2)
		line_updated = line_part1 + updated_part1 + line_part2

	lines[line_idx] = line_updated
	# print (line_updated)



######################################## MODIFY XML FUNCTION ########################################
def modify_xml(robot_name, parameters=None, directory_path=None):
	if directory_path == None:
		directory_path = '/home/yetong/Desktop/roboschool-master/roboschool/mujoco_assets'
	file_original = robot_name + '_original.xml'
	file_new = robot_name + '.xml'

	with open (os.path.join(directory_path, file_original), 'r') as f:
		lines = f.read().splitlines()

	if 'ant' in file_new:
		if parameters == None:
			parameters = [0.2, 0.08, 0.4, 0.08] * 4 # length1, rad1, length2, rad2 for each link

		starting_lines = [14, 25, 36, 47]
		directions = [(1,1), (-1,1), (-1,-1), (1,-1)]
		for idx, starting_line in enumerate(starting_lines):
			len_1, rad_1, len_2, rad_2 = parameters[idx*4: (idx+1)*4]
			# update_line(lines, starting_line+1, len_1, rad_1, directions[idx], mode=1)
			# update_line(lines, starting_line+2, len_1, rad_1, directions[idx], mode=2)
			update_line_ant(lines, starting_line+4, len_1, rad_1, directions[idx], mode=1)
			update_line_ant(lines, starting_line+5, len_1, rad_1, directions[idx], mode=2)
			update_line_ant(lines, starting_line+7, len_2, rad_2, directions[idx], mode=1)

	if 'hopper' in file_new:
		if parameters == None:
			parameters = [0.4, 0.05, 0.45, 0.05, 0.5, 0.04, 0.13, 0.06, 0.26, 0.06] # length and rad for torso, thigh, leg, foot1, foot2
		starting_line = 14
		len_torso, rad_torso, len_thigh, rad_thigh, len_leg, rad_leg, len_foot1, rad_foot1, len_foot2, rad_foot2 = parameters
		z_foot = 0.1
		z_leg = z_foot+len_leg
		z_thigh = z_leg + len_thigh
		z_torso = z_thigh + len_torso

		update_line_hopper(lines, starting_line+4, 0, z_torso, 0, z_thigh, rad_torso, mode=1) # update torso
		update_line_hopper(lines, starting_line+6, 0, z_torso, 0, z_thigh, rad_torso, mode=2) # update thigh axis
		update_line_hopper(lines, starting_line+7, 0, z_thigh, 0, z_leg, rad_thigh, mode=1) # update thigh
		update_line_hopper(lines, starting_line+9, 0, z_thigh, 0, z_leg, rad_thigh, mode=2) # update leg axis
		update_line_hopper(lines, starting_line+10, 0, z_leg, 0, z_foot, rad_leg, mode=1) # update leg
		update_line_hopper(lines, starting_line+12, 0, z_leg, 0, z_foot, rad_leg, mode=2) # update foot axis	
		update_line_hopper(lines, starting_line+13, -len_foot1, z_foot, len_foot2, z_foot, rad_foot1, mode=1) # update foot		

	if 'walker2d' in file_new:
		if parameters == None:
			parameters = [0.4, 0.05, 0.45, 0.05, 0.5, 0.04, 0.2, 0.06] # length and rad for torso, thigh, leg, foot
		len_torso, rad_torso, len_thigh, rad_thigh, len_leg, rad_leg, len_foot, rad_foot = parameters
		z_foot = 0.1
		z_leg = z_foot+len_leg
		z_thigh = z_leg + len_thigh
		z_torso = z_thigh + len_torso

		update_line_hopper(lines, 13, 0, z_torso, 0, z_thigh, rad_torso, mode=1) # update torso
		starting_lines = [14, 27]
		for idx, starting_line in enumerate(starting_lines):
			update_line_hopper(lines, starting_line+1, 0, z_torso, 0, z_thigh, rad_torso, mode=2) # update thigh axis
			update_line_hopper(lines, starting_line+2, 0, z_thigh, 0, z_leg, rad_thigh, mode=1) # update thigh
			update_line_hopper(lines, starting_line+4, 0, z_thigh, 0, z_leg, rad_thigh, mode=2) # update leg axis
			update_line_hopper(lines, starting_line+5, 0, z_leg, 0, z_foot, rad_leg, mode=1) # update leg
			update_line_hopper(lines, starting_line+7, 0, z_leg, 0, z_foot, rad_leg, mode=2) # update foot axis	
			update_line_hopper(lines, starting_line+8, 0, z_foot, len_foot, z_foot, rad_foot, mode=1) # update foot		


	with open (os.path.join(directory_path, file_new), 'w') as f:
		for line in lines:
			f.write(line+'\n')

	print ('Environment changed!')

def main():
	modify_xml('ant')

if __name__ == '__main__':
	main()