



def get_largest_iteration(directory):
	try:
		largest_path = None
		largest_iteration = -1
		for filename in os.listdir(directory):
			m = ITERATION_PATTERN.match(filename)
			if not m:
				continue
			if largest_iteration is None or int(m.group(1)) > largest_iteration:
				largest_path = filename
				largest_iteration = int(m.group(1))
		return largest_path, largest_iteration
	except:
		print('unable to find largest iteration within ' + directory)
		return None, None


def get_run_result(itdir, table_entry):
	largest_path, table_entry.num_iterations = get_largest_iteration(itdir)
	if largest_path is None:
		return None, None, None, None

	with open(os.path.join(itdir, largest_path)) as iteration_input:
		iteration_info = json.load(iteration_input)
		table_entry.success_count = 0
		table_entry.fail_count = 0

		for _, evaluation in iteration_info['state']['history']['evaluations']:
			table_entry.accept(Evaluation.parse_json(evaluation))

		if 'convexity' in iteration_info['state']:
			convexity_test = ConvexityTester.parse_json(iteration_info['state']['convexity'])
			table_entry.definiteness = str(convexity_test)
		else:
			table_entry.definiteness = 'unknown'

		it_result = iteration_info['iteration-result']
		table_entry.result = it_result['converged'] if it_result is not None else None


# def get_number_of_evaluations(problem_dir):
# 	try:
# 		hit_history_begin = 0
# 		success_count = 0
# 		fail_count = 0
# 		for line in open(os.path.join(problem_dir, 'log_file.txt')):
# 			if line == '[] history\n':
# 				hit_history_begin = 0
# 			elif hit_history_begin == 0 and line == '[] {\n':
# 				hit_history_begin = 1
# 			elif hit_history_begin == 1:
# 				if '"success": true' in line:
# 					success_count += 1
# 				elif '"success": false' in line:
# 					fail_count += 1
# 		return True, success_count, fail_count
# 	except:
# 		print('unable to count evaluations from ' + itdir)
# 		return False, None, None

