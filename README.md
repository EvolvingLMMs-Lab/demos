---
title: LLaVA
app_file: serve/gradio_web_server.py
sdk: gradio
sdk_version: 4.29.0
---
# Installation

<!-- curl -L --output cloudflared.deb https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb &&  -->

sudo dpkg -i llava_demo.deb && 

sudo cloudflared service install eyJhIjoiZjMyZGYyNTNmMzVjMzA5ODBjZTMyMGM0MTUyZjZjZmEiLCJ0IjoiYTM5MzMzYWYtYTY3OC00MThlLWIyZWYtZDQzNmVkNDRhMzc0IiwicyI6Ik1qZ3laalk1TnpRdE1EQmlNeTAwWkRjM0xXRTFaR1F0WXpCbU9XSTNOelZtTmpNMyJ9

prepare_demo;
build_llava;

cd sglang_codebase;
pip install -e "python[srt]"

cd ..;

CUDA_VISIBLE_DEVICES=4,5,6,7 python -m sglang.launch_server --model-path lmms-lab/llava-next-72b --tokenizer-path lmms-lab/llavanext-qwen-tokenizer --port=30000 --host="127.0.0.1" --tp-size=4

python -m sglang.launch_server --model-path lmms-lab/llava-next-72b --tokenizer-path lmms-lab/llavanext-qwen-tokenizer --port=30000 --host="127.0.0.1" --tp-size=8

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m sglang.launch_server --model-path lmms-lab/llama3-llava-next-8b --tokenizer-path lmms-lab/llama3-llava-next-8b-tokenizer --port=10000 --host="127.0.0.1" --tp-size=4


python serve/controller.py --host 0.0.0.0 --port 12355
python serve/gradio_web_server.py --controller-url=http://localhost:12355 --model-list-mode reload --moderate --share
python serve/sglang_worker.py --host 0.0.0.0 --controller http://localhost:12355 --port 3005 --worker http://localhost:3005 --sgl-endpoint http://127.0.0.1:30000
python serve/sglang_worker.py --host 0.0.0.0 --controller http://localhost:12355 --port 3000 --worker http://localhost:3000 --sgl-endpoint http://127.0.0.1:10000
<!-- python multimodal_chat.py --sglang_port=30000 -->

pip install gradio==4.29.0
pip install httpx==0.23.3