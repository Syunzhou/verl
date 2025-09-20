set -x
export HYDRA_FULL_ERROR=1

nproc_per_node=1
save_path="../../../verl_details/sft/test/"
data_dir="../../../download_data/gsm8k/sft_data"

model_name="qwen3_06b"
pretrain_model=../../../download_models/$model_name

torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
     -m verl.trainer.fsdp_sft_trainer \
    data.train_files=$data_dir/train.parquet \
    data.val_files=$data_dir/test.parquet \
    data.prompt_key=extra_info \
    data.response_key=extra_info \
    data.prompt_dict_keys=['question'] \
    +data.response_dict_keys=['answer'] \
    data.micro_batch_size_per_gpu=4 \
    model.partial_pretrain=$pretrain_model \
    trainer.default_local_dir=$save_path \
    trainer.project_name=gsm8k-sft \
    trainer.experiment_name=gsm8k-sft-$model_name \
    trainer.total_epochs=2 \
    trainer.logger='["console","tensorboard"]' $@