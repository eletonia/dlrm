#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
#WARNING: must have compiled PyTorch and caffe2

#check if extra argument is passed to the test
if [[ $# == 1 ]]; then
    dlrm_extra_option=$1
else
    dlrm_extra_option=""
fi
#echo $dlrm_extra_option

dlrm_pt_bin="python3 dlrm_s_pytorch.py"
dlrm_c2_bin="python3 dlrm_s_caffe2.py"

echo "run pytorch ..."
# WARNING: the following parameters will be set based on the data set
# --arch-embedding-size=... (sparse feature sizes)
# --arch-mlp-bot=... (the input to the first layer of bottom mlp)

#Add nsys profiling and run DLRM inference only code
export GPU_ID=4 #choose a device id 
set -x
CUDA_VISIBLE_DEVICES=$GPU_ID
nsys profile -f true -o timeline -c cudaProfilerApi --stop-on-range-end true -s none --gpu-metrics-device=$GPU_ID --gpu-metrics-set=ga100\
 $dlrm_pt_bin --arch-sparse-feature-size=16 \
	--arch-mlp-bot="13-512-256-64-16" \
	--arch-mlp-top="512-256-1" \
	--data-generation=dataset \
	--data-set=kaggle \
	--raw-data-file=/mnt/nvme1n1p1/DLRM/data-medium/train.txt \
	--processed-data-file=/mnt/nvme1n1p1/DLRM/data-medium/kaggleAdDisplayChallenge_processed.npz \
	--inference-only \
	--loss-function=bce \
	--round-targets=True \
	--load-model=/mnt/nvme1n1p1/DLRM/data-medium/criteo-medium-100bat.pt \
	--print-freq=1024 \
	--print-time \
	--test-mini-batch-size=16384 \
	--num-batches=65536 \
	--test-num-workers=8 \
	--use-gpu \
	$dlrm_extra_option 2>&1 | tee inf_kaggle_pt.log

# echo "run caffe2 ..."
# # WARNING: the following parameters will be set based on the data set
# # --arch-embedding-size=... (sparse feature sizes)
# # --arch-mlp-bot=... (the input to the first layer of bottom mlp)
# $dlrm_c2_bin --arch-sparse-feature-size=16 --arch-mlp-bot="13-512-256-64-16" --arch-mlp-top="512-256-1" --data-generation=dataset --data-set=kaggle --raw-data-file=./input/train.txt --processed-data-file=./input/kaggleAdDisplayChallenge_processed.npz --loss-function=bce --round-targets=True --learning-rate=0.1 --mini-batch-size=128 --print-freq=1024 --print-time $dlrm_extra_option 2>&1 | tee run_kaggle_c2.log

echo "done"
