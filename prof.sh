#!/bin/bash

# This script is the helper script for launching nvidia's nsight systems
# profiling tool for your application. The most important assumptions to 
# be aware of are:
# 1. Single GPU. Multi GPU support is future work
# 2. The application has the proper instrumentation to start profiling
#    and stop profiling. See https://github.ibm.com/hybrid-cloud-infrastructure-research/tracker/issues/257#issuecomment-41770107
# 3. Only tested on A100 machine with 8 GPUs and 2 CPU sockets. CORES
#    range calculation for taskset needs modifications for other machines

PROFILER=${PROFILER:-nsys}

# Currently we assume only using single GPU
# Note that this is a list of IDs, not the number of GPUS visible
export GPU_ID=4 #choose a device id 
CUDA_VISIBLE_DEVICES=$GPU_ID

# Change GPU type for nsight systems to collect the right metrics
GPU_TYPE=${GPU_TYPE:-ga100}
GPU_LIBS_TRACE=${GPU_LIBS_TRACE:-cuda,cublas,cudnn}

# Set the app name. If $APP is set in the environment, that value will be used.
# Otherwise the default value "xquad" is used.
APP="dlrm"

# Your application lauch command. Customize as needed
APP_LAUNCH_CMD_FILE=${APP_LAUNCH_CMD_FILE:-launch-${APP}}
APP_LAUNCH_CMD=${APP_LAUNCH_CMD:-$(<${APP_LAUNCH_CMD_FILE})}

# Uncomment the following to use launch command inline
#APP_LAUNCH_CMD="
#python third_party/run_squad.py 
#    --model_type xlm-roberta 
#    --model_name_or_path outputs/xquad/xlm-roberta_SEED42_LR3e-5_EPOCH1_maxlen384 
#    --do_eval --do_lower_case --eval_lang en 
#    --predict_file /nvme/ccyang/xquad/xquad.en.json 
#    --output_dir predictions/xquad_SEED42/"

TIME_STAMP=$(date +%s)

# Note: the CORES range limits the scheduling of app process
#       to a specific socket. Change this according to your
#       machine setup
if (( $CUDA_VISIBLE_DEVICES < 4 )); then
    CORES="0-23"
else
    CORES="24-47"
fi

if [ ${PROFILER} != "ncu" ]; then  
    # Nsight Systems profiling
    NSYS_BIN=${NSYS_BIN:-/usr/local/cuda/bin/nsys}
    if [ ! -f "$NSYS_BIN" ]; then
        echo $NSYS_BIN does not exist!
        exit -1
    fi

    set -x
    taskset -c ${CORES} \
        ${NSYS_BIN} profile \
        -f true -o ${APP}-${TIME_STAMP}-timeline -c cudaProfilerApi \
        --capture-range-end=stop --gpu-metrics-device=${CUDA_VISIBLE_DEVICES}\
        --gpu-metrics-set=${GPU_TYPE} --trace=${GPU_LIBS_TRACE}\
        ${APP_LAUNCH_CMD} 2>&1 | tee ${APP}-${TIME_STAMP}.log.txt
else 
    # Nsight Compute profiling
    NCU_BIN=${NCU_BIN:-/usr/local/cuda/bin/ncu}
    if [ ! -f "$NCU_BIN" ]; then
        echo $NCU_BIN does not exist!
        exit -1
    fi

    set -x
    taskset -c ${CORES} \
        ${NCU_BIN} --profile-from-start off  \
        -o ${APP}-${TIME_STAMP}-profile \
        --kernel-name-base mangled \
        -k regex:_ZN7cutlass6KernelI52cutlass_80_tensorop_s1688gemm_ -s 200 -c 12 \
        --section SpeedOfLight \
        --section ComputeWorkloadAnalysis \
        --section MemoryWorkloadAnalysis \
        --section Occupancy \
        --section SchedulerStats \
        --section WarpStateStats \
        ${APP_LAUNCH_CMD} 2>&1 | tee ${APP}-${TIME_STAMP}.log.txt
fi
