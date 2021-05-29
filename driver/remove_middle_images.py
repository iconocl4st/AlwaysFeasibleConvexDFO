import os
import time


def count_disk_usage(directory):
	return sum(
		os.stat(filepath).st_size
		for filepath in (
			os.path.join(directory, filename)
			for filename in os.listdir(directory))
		if os.path.isfile(filepath))


def remove_middle_files(directory, dry_run=True, num_to_keep=20):
	for filepath, _ in sorted((
		(filepath, numeral) for filepath, numeral in (
			(os.path.join(directory, filename), int(''.join(c for c in filename if '0' <= c <= '9')))
		for filename in os.listdir(directory))
		if os.path.isfile(filepath)),
		key=lambda x: x[1])[num_to_keep:-num_to_keep]:
		if dry_run:
			print('rm', filepath)
		else:
			os.remove(filepath)
	print(count_disk_usage(directory))


def prune_large_directories(root_directory):
	gigabyte = 1024 * 1024 * 1024
	for root, directories, _ in os.walk(root_directory):
		for directory in directories:
			child_path = os.path.join(root, directory)
			directory_size = count_disk_usage(child_path)
			if directory_size < 10 * gigabyte:
				continue
			print(str(round(directory_size / gigabyte, 2)) + " GB", child_path)
			remove_middle_files(child_path, dry_run=False, num_to_keep=30)


def repeatedly_prune(root_directory, sleep_time):
	while True:
		print('Pruning...')
		prune_large_directories(root_directory)
		print('...done.')
		time.sleep(sleep_time)


if __name__ == '__main__':
	repeatedly_prune(
		'/home/thallock/Pictures/ConvexConstraintsOutput/runs/',
		10 * 60)

