#!/bin/bash

# This script is the helper script for launching nvidia's nsight systems
# profiling tool for your application. The most important assumptions to 
# be aware of are:
# 1. Single GPU and Multi GPU support (Thanks to @maurizio-drocco)
# 2. The application has the proper instrumentation to start profiling
#    and stop profiling. See https://github.ibm.com/hybrid-cloud-infrastructure-research/tracker/issues/257#issuecomment-41770107
#    You can now set $NSYS_PROF_RANGE to "app" (default: "api") if you want to profile the whole app
#    and in that case, no need to instrument the code.
# 3. This was tested on A100 machine with 8 GPUs and 2 CPU sockets. CORES
#    range calculation is new

PROFILER=${PROFILER:-nsys}

# Set NSYS_PROF_RANGE to select the range for nsys profiling:
# - "api": enabled by CUDA API calls
# - "app": whole app
NSYS_PROF_RANGE=${NSYS_PROF_RANGE:-"api"}

# Currently we assume only using single GPU
# Note that this is a list of IDs, not the number of GPUS visible
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# Change CORES_PER_GPU according to your machine setup
CORES_PER_GPU=${CORES_PER_GPU:-6}

# Change GPU type for nsight systems to collect the right metrics
GPU_TYPE=${GPU_TYPE:-v100}
GPU_LIBS_TRACE=${GPU_LIBS_TRACE:-cuda,cublas,cudnn}

# Set the app name. If $APP is set in the environment, that value will be used.
# Otherwise the default value "xquad" is used.
APP=${APP:-xquad}

# Your application lauch command. Customize as needed
APP_LAUNCH_SCRIPT=${APP_LAUNCH_SCRIPT:-launch-${APP}.sh}
APP_LAUNCH_CMD=${APP_LAUNCH_CMD:-$(<${APP_LAUNCH_SCRIPT})}

# Specify the folder you want the profile and log to be
PROF_OUT_DIR=${PROF_OUT_DIR:-${PWD}}
mkdir -p $PROF_OUT_DIR

TIME_STAMP=$(date +%s)

# Core range calculation
CUDA_FIRST_DEVICE=$(echo $CUDA_VISIBLE_DEVICES | cut -d , -f 1)
CUDA_TAIL_DEVICES=$(echo $CUDA_VISIBLE_DEVICES | sed 's/[0-9]\+,\?//')
CORE_FIRST=$(( CUDA_FIRST_DEVICE * CORES_PER_GPU ))
CORE_LAST=$(( (CUDA_FIRST_DEVICE + 1) * CORES_PER_GPU - 1))
for gpu in ${CUDA_TAIL_DEVICES//,/ }
do
    CORE_GPU_FIRST=$(( gpu * CORES_PER_GPU ))
    CORE_GPU_LAST=$(( (gpu + 1) * CORES_PER_GPU - 1))
    if [ $CORE_GPU_FIRST -lt $CORE_FIRST ]; then CORE_FIRST=$CORE_GPU_FIRST; fi
    if [ $CORE_GPU_LAST -gt $CORE_LAST ]; then CORE_LAST=$CORE_GPU_LAST; fi
done
# Note: the CORES range binds process to specific cores
#       ideally they should be the same socket close to the gpu
#       used. Override it manually if you want specific setting
#       for your process
CORES=${CORES:-"$CORE_FIRST-$CORE_LAST"}


if [ ${PROFILER} != "ncu" ]; then  
    # Nsight Systems profiling
    NSYS_BIN=${NSYS_BIN:-/usr/local/cuda/bin/nsys}
    if [ ! -f "$NSYS_BIN" ]; then
        echo $NSYS_BIN does not exist!
        exit -1
    fi

    case $NSYS_PROF_RANGE in
        api)
            NSYS_PROF_RANGE_OPTS="-c cudaProfilerApi --capture-range-end=stop"
            ;;
        app)
            NSYS_PROF_RANGE_OPTS=""
            ;;
    esac

    # Only A100 supports gpu-metrics for now
    case $GPU_TYPE in
        ga100)
            NSYS_GPU_METRICS_OPTS="--gpu-metrics-device=${CUDA_VISIBLE_DEVICES} --gpu-metrics-set=${GPU_TYPE}"
            ;;
        *)
            NSYS_GPU_METRICS_OPTS=""
            ;;
    esac

    # nsys will accept a list of devices (as in CUDA_VISIBLE_DEVICES)
    # despite the documentation says it only allows either a single ID or 'all'
    set -x
    taskset -c ${CORES} \
        ${NSYS_BIN} profile \
        -f true -o ${PROF_OUT_DIR}/${APP}-${TIME_STAMP}-timeline \
        ${NSYS_PROF_RANGE_OPTS} \
        ${NSYS_GPU_METRICS_OPTS} \
        --trace=${GPU_LIBS_TRACE} \
        bash -c "${APP_LAUNCH_CMD}" 2>&1 | tee ${PROF_OUT_DIR}/${APP}-${TIME_STAMP}.log.txt
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
        -o ${PROF_OUT_DIR}/${APP}-${TIME_STAMP}-profile \
	--kernel-name-base mangled \
	-k regex:gemm -s 200 -c 12 \
        --section SpeedOfLight \
    	--section SpeedOfLight_RooflineChart \
        --section ComputeWorkloadAnalysis \
        --section MemoryWorkloadAnalysis \
        --section Occupancy \
        --section SchedulerStats \
        --section WarpStateStats \
	--target-processes all \
        bash -c "${APP_LAUNCH_CMD}" 2>&1 | tee ${PROF_OUT_DIR}/${APP}-${TIME_STAMP}.log.txt
fi
