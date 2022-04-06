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
	--test-mini-batch-size=16384 \
	--num-batches=65536 \
	--test-num-workers=8 \
	--use-gpu \
	$dlrm_extra_option 2>&1 | tee inf_kaggle_pt.log

