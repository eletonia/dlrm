#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
#WARNING: must have compiled PyTorch and caffe2

#check if extra argument is passed to the test
NumBatches=${1:-20}
NumWorkers=${2:-16}
BatchSize=${3:-1024}
EnableProfile=${4:-"NOPROFILE"}
DoRamDisk=${5:-"ramdisk"}
DoMultiProc=${6:-"MULTI"}
dlrm_extra_option=${7:-""}
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1
#echo $dlrm_extra_option

enable_profiling=" --enable-profiling "
if [[ $EnableProfile == "NOPROFILE" ]]; then
	enable_profiling=""
fi

basedir=/nvme/DLRM/data-medium
if [[ $DoRamDisk == "ramdisk" ]]; then
	basedir=/tmp/dlrm_rd
fi

dataset_multiprocessing=""
if [[ $DoMultiProc == "MULTI" ]]; then
	dataset_multiprocessing=" --dataset-multiprocessing "
fi

dlrm_pt_bin="python3 dlrm_s_pytorch.py"
dlrm_c2_bin="python3 dlrm_s_caffe2.py"

echo "run pytorch ... "
printf "NumWorkers: %d , BatchSize: %d, Multiproc: %s\n" $NumWorkers $BatchSize $dataset_multiprocessing

# WARNING: the following parameters will be set based on the data set
# --arch-embedding-size=... (sparse feature sizes)
# --arch-mlp-bot=... (the input to the first layer of bottom mlp)
$dlrm_pt_bin --arch-sparse-feature-size=16 \
	--arch-mlp-bot="13-512-256-64-16" \
	--arch-mlp-top="512-256-1" \
	--data-generation=dataset \
	--data-set=kaggle \
	--raw-data-file=$basedir/train.txt \
	--processed-data-file=$basedir/kaggleAdDisplayChallenge_processed.npz \
	--inference-only \
	--loss-function=bce \
	--round-targets=True \
	--load-model=$basedir/criteo-medium-100bat.pt \
	--print-freq=1024 \
	--print-time $dataset_multiprocessing $enable_profiling \
	--data-randomize=none \
	--mini-batch-size=$BatchSize \
	--test-mini-batch-size=$BatchSize \
	--num-batches=$NumBatches \
	--test-num-workers=$NumWorkers \
	--num-workers=$NumWorkers \
	--memory-map \
	--use-gpu \
	$dlrm_extra_option 
	
	# 2>&1 | tee inf_kaggle_pt.log

	# --dataset-multiprocessing \
	# --use-gpu \

# echo "run caffe2 ..."
# # WARNING: the following parameters will be set based on the data set
# # --arch-embedding-size=... (sparse feature sizes)
# # --arch-mlp-bot=... (the input to the first layer of bottom mlp)
# $dlrm_c2_bin --arch-sparse-feature-size=16 --arch-mlp-bot="13-512-256-64-16" --arch-mlp-top="512-256-1" --data-generation=dataset --data-set=kaggle --raw-data-file=./input/train.txt --processed-data-file=./input/kaggleAdDisplayChallenge_processed.npz --loss-function=bce --round-targets=True --learning-rate=0.1 --mini-batch-size=128 --print-freq=1024 --print-time $dlrm_extra_option 2>&1 | tee run_kaggle_c2.log

echo "done"
