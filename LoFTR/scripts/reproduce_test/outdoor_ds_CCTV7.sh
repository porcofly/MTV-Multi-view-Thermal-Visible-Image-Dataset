#!/bin/bash -l

SCRIPTPATH=$(dirname $(readlink -f "$0"))
PROJECT_DIR="${SCRIPTPATH}/../../"

# conda activate loftr
export PYTHONPATH=$PROJECT_DIR:$PYTHONPATH
cd $PROJECT_DIR

data_cfg_path="configs/data/CCTV7_test.py"
main_cfg_path="configs/loftr/outdoor/buggy_pos_enc/loftr_ds_dense.py"
ckpt_path="logs/tb_logs/outdoor-ds-640-bs=4/version_4/checkpoints/epoch=231-auc@5=0.056-auc@10=0.134-auc@20=0.230.ckpt"
dump_dir="dump/best_loftr"
profiler_name="inference"
n_nodes=1  # mannually keep this the same with --nodes
n_gpus_per_node=2
torch_num_workers=4
batch_size=16 # per gpu

CUDA_VISIBEL_DEVICES=1 python -u ./test.py \
    ${data_cfg_path} \
    ${main_cfg_path} \
    --ckpt_path=${ckpt_path} \
    --dump_dir=${dump_dir} \
    --gpus=${n_gpus_per_node} --num_nodes=${n_nodes} --accelerator="ddp" \
    --batch_size=${batch_size} --num_workers=${torch_num_workers}\
    --profiler_name=${profiler_name} \
    --benchmark 
    