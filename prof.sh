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
export CUDA_AUTO_BOOST=1 # Enable auto boost by default for performance profiling

# Change CORES_PER_GPU according to your machine setup
CORES_PER_GPU=${CORES_PER_GPU:-6}

# Change GPU type for nsight systems to collect the right metrics
GPU_METRICS_TYPE=${GPU_METRICS_TYPE:-ga100}
GPU_LIBS_TRACE=${GPU_LIBS_TRACE:-cuda,cublas,cudnn}

# Set the app name. If $APP is set in the environment, that value will be used.
# Otherwise the default value "xquad" is used.
APP=${APP:-xquad}

# Your application lauch command. Customize as needed
APP_LAUNCH_SCRIPT=${APP_LAUNCH_SCRIPT:-launch-${APP}.sh}
APP_LAUNCH_CMD=${APP_LAUNCH_CMD:-$(<${APP_LAUNCH_SCRIPT})}

NCU_KERNEL_REGEX=${NCU_KERNEL_REGEX:-gemm}

# Specify the folder you want the profile and log to be
PROF_OUT_DIR=${PROF_OUT_DIR:-${PWD}}
mkdir -p $PROF_OUT_DIR

TIME_STAMP=$(date +%s)

# ############################################
# BEGIN block for core range calculation
# ############################################
AFFINITY_FIELD_IDX=""

function field_name_at() {
    local field_cnt=$1
    nvidia-smi topo -m | head -n 1 | awk -F"\t" "{print \$${field_cnt}}" | egrep "[[:print:]]+"
}

function affinity() {
    local gpu=$1
    nvidia-smi topo -m | egrep "^GPU${gpu}\>" | awk -F"\t" "{print \$${AFFINITY_FIELD_IDX}}" | egrep "[[:print:]]+"
}

function cores() {
    local cores=""
    local field_cnt=2

    while field_name_at $field_cnt > /dev/null
    do
        if field_name_at $field_cnt | grep -q "CPU Affinity"
        then
            AFFINITY_FIELD_IDX=$field_cnt
            break
        fi
        field_cnt=$(( field_cnt + 1 ))
    done

    if [ ! -z $AFFINITY_FIELD_IDX ]
    then
        for gpu in ${CUDA_VISIBLE_DEVICES//,/ }
        do
            ga=$(affinity $gpu)
            if [ $? -eq 0 ]
            then
                if [[ "$cores" == *"$ga"* ]]
                then
                    true
                else
                    cores="$cores,$ga"
                fi
            else
                return 2
            fi
        done
    else
        return 1
    fi

    if [ ! -z $cores ]
    then
        cores=$(echo $cores | sed 's/^,//')
    else
        return 3
    fi

    echo $cores
}
# ############################################
# END block for core range calculation
# ############################################

# Note: the CORES range binds process to specific cores
#       ideally they should be the same socket close to the gpu
#       used. Override it manually if you want specific setting
#       for your process
if [ -z $CORES ]
then
    CORES=$(cores)
    ret=$?
    if [ $ret -ne 0 ]
    then
        echo "ERROR: core range calculation failed (err=$ret)"
        exit 1
    fi
fi

# Compute what options to use depending on whether whole application is profiled
case $NSYS_PROF_RANGE in
    api)
        NSYS_PROF_RANGE_OPTS="-c cudaProfilerApi --capture-range-end=stop"
        NCU_PROF_FROM_START="off"
        ;;
    app)
        NSYS_PROF_RANGE_OPTS=""
        NCU_PROF_FROM_START="on"
        ;;
esac

# Check for supported metrics types
# Will not generate gpu metrics if the device doesn't support it
case $GPU_METRICS_TYPE in
    tu10x|tu11x|ga100|ga10x|tu10x-gfxt|ga10x-gfxt|ga10x-gfxact)
        NSYS_GPU_METRICS_OPTS="--gpu-metrics-device=2 --gpu-metrics-set=${GPU_METRICS_TYPE}"
        ;;
    *)
        NSYS_GPU_METRICS_OPTS=""
        ;;
esac

if [ ${PROFILER} != "ncu" ]; then  
    # Nsight Systems profiling
    NSYS_BIN=${NSYS_BIN:-/usr/local/cuda/bin/nsys}
    if [ ! -f "$NSYS_BIN" ]; then
        echo $NSYS_BIN does not exist!
        exit -1
    fi

    # nsys will accept a list of devices (as in CUDA_VISIBLE_DEVICES)
    # despite the documentation says it only allows either a single ID or 'all'
    set -x
    taskset -c ${CORES} \
        ${NSYS_BIN} profile \
        -f true -o ${PROF_OUT_DIR}/${APP}-${TIME_STAMP}-timeline \
        ${NSYS_PROF_RANGE_OPTS} \
        ${NSYS_GPU_METRICS_OPTS} \
        --trace=${GPU_LIBS_TRACE} \
	#--gpu-metrics-frequency=100000 \
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
        ${NCU_BIN} --profile-from-start ${NCU_PROF_FROM_START} \
        -o ${PROF_OUT_DIR}/${APP}-${TIME_STAMP}-profile \
        --kernel-name-base mangled \
        -k regex:${NCU_KERNEL_REGEX} -s 200 -c 12 \
        --section SpeedOfLight \
        --section ComputeWorkloadAnalysis \
        --section MemoryWorkloadAnalysis \
        --section Occupancy \
        --section SchedulerStats \
        --section WarpStateStats \
        --target-processes all \
        bash -c "${APP_LAUNCH_CMD}" 2>&1 | tee ${PROF_OUT_DIR}/${APP}-${TIME_STAMP}.log.txt
fi
