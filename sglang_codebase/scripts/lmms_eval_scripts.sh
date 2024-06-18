CKPT_PATH="/mnt/bn/vl-research/checkpoints/llavanext-openai_clip-vit-large-patch14-336-meta-llama_Meta-Llama-3-8B-Instruct-blip558k_pretrain_plain_unmask_special_tokens"
TOK_PATH="lmms-lab/llama3-llava-next-8b-tokenizer"
#CKPT_PATH="/mnt/bn/vl-research-boli01-cn/workspace/zzz/projects/zzz/llava_next/project_checkpoints/llava_vicuna_clip_blip558_pretrained_sampled"
CHAT_TEMPLATE="llama-3-instruct"
#CKPT_PATH="liuhaotian/llava-v1.6-34b"
python3 -m lmms_eval \
    --model llava_sglang \
    --model_args pretrained=${CKPT_PATH},tokenizer=${TOK_PATH},conv_template=${CHAT_TEMPLATE},tp_size=8,parallel=8 \
    --tasks mme \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix llava_llama3 \
    --output_path ./logs/