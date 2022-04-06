python3 dlrm_s_pytorch.py \
	--arch-sparse-feature-size=16 \
	--arch-mlp-bot=13-512-256-64-16 \
	--arch-mlp-top=512-256-1 \
	--data-generation=dataset \
	--data-set=kaggle \
	--raw-data-file=/tmp/dlrm_rd/train.txt \
	--processed-data-file=/tmp/dlrm_rd/kaggleAdDisplayChallenge_processed.npz \
	--inference-only \
	--loss-function=bce \
	--round-targets=True \
	--load-model=/tmp/dlrm_rd/criteo-medium-100bat.pt \
	--print-freq=1024 \
	--print-time \
	--mini-batch-size=$BatchSize \
	--test-mini-batch-size=$BatchSize \
	--num-batches=$NumBatches \
	--test-num-workers=$NumWorkers \
	--num-workers=$NumWorkers \
	--use-gpu \
	$dlrm_extra_option 2>&1 | tee inf_kaggle_pt.log

