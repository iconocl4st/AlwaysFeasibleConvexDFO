#!/bin/bash

# fail-with-no-information fail-with-information no-failures

for problem_no in 12 24 29 30 31 33 34 35 36 37 43 44 57 66 67 76 84 86 93 100 105 215 218 221 223 224 226 227 228 231 232 233 249 250 251 253 264 268 270 315 323 326 329 331 337 339 341 342 354 359
do
	for strategy in fail-with-garbage
	do
		PYTHONUNBUFFERED=1 \
			CONDA_PREFIX=/media/thallock/1f0ab4b3-c472-49e1-92d8-c0b5664f7fdb/anaconda3/envs/ConvexConstraints \
			CONDA_DEFAULT_ENV=ConvexConstraints \
			CONDA_PREFIX_1=/media/thallock/1f0ab4b3-c472-49e1-92d8-c0b5664f7fdb/anaconda3 \
			PATH=/media/thallock/1f0ab4b3-c472-49e1-92d8-c0b5664f7fdb/anaconda3/envs/ConvexConstraints/bin:$PATH \
			PYTHONPATH=/home/thallock/PycharmProjects/ConvexConstraints:$PYTHONPATH \
			/mnt/1f0ab4b3-c472-49e1-92d8-c0b5664f7fdb/anaconda3/envs/ConvexConstraints/bin/python ./driver/run_nomad.py $problem_no $strategy
	done
done


