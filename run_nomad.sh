#!/bin/bash


for problem_no in 201 202 203 204 205 206 207 208 209 210 211 212 213 214 215 216 217 218 219 220 221 222 223 224 225 226 227 228 229 230 231 232 233 234 235 236 237 238 239 240 241 242 243 244 244 245 246 248 249 250 251 252 253 254 255 256 257 258 259 260 261 262 263 264 265 268 270 284 285 315 323 326 332 342 353 359 1 12 24 29 30 31 33 34 35 36 37 43 44 57 59 66 67 71 76 84 86 93 100 105 110 113 117 118
do
	for strategy in fail-with-no-information fail-with-garbage fail-with-information no-failures
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


