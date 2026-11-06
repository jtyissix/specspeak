#set -e
#set -x

######################################################################
#export NCCL_NET=IB

#export NCCL_SOCKET_IFNAME="bond1"
#export GLOO_SOCKET_IFNAME="bond1"
#export NCCL_DEBUG=INFO
#export NCCL_IB_QPS_PER_CONNECTION=2

#export GLOO_SOCKET_IFNAME=eth0
#export NCCL_DEBUG=INFO
#export NCCL_IB_QPS_PER_CONNECTION=2

#export NCCL_IB_DISABLE=1

export DISTRIBUTED_BACKEND="nccl"
export CUDA_DEVICE_MAX_CONNECTIONS=1

######################################################################
#pip install -r requirements_ds_gpu.txt
#pip3 install --no-index --find-links=${ROOT_PATH}/software/ -r requirements_ds_gpu.txt

#pip install deepspeed==0.15.4
#pip3 install --no-index --find-links=${ROOT_PATH}/software/ deepspeed==0.15.4
#pip3 install deepspeed==0.16.1
#pip3 install deepspeed==0.14.2
VITA_PROJECT_ROOT="${CODE_PATH:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
export PYTHONPATH="${VITA_PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

#pip install -e `pwd`

######################################################################
#export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

#apt update
#apt install -y openssh-server rsync tmux htop

######################################################################

export NNODES=${WORLD_SIZE}
export NODE_RANK=${RANK}
export MASTER_PORT=34567
if [ -z "$NPROC_PER_NODE" ]
then
    export NPROC_PER_NODE=4
    export NNODES=1
    export NODE_RANK=0
    export MASTER_ADDR=127.0.0.1
fi

######################################################################
