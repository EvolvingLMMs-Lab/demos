# Installation

<!-- curl -L --output cloudflared.deb https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb &&  -->

sudo dpkg -i cloudflared.deb && 

sudo cloudflared service install eyJhIjoiZjMyZGYyNTNmMzVjMzA5ODBjZTMyMGM0MTUyZjZjZmEiLCJ0IjoiYTM5MzMzYWYtYTY3OC00MThlLWIyZWYtZDQzNmVkNDRhMzc0IiwicyI6Ik1qZ3laalk1TnpRdE1EQmlNeTAwWkRjM0xXRTFaR1F0WXpCbU9XSTNOelZtTmpNMyJ9

cd sglang_codebase;
pip install -e "python[srt]"

cd ..;

CUDA_VISIBLE_DEVICES=4,5,6,7 python -m sglang.launch_server --model-path lmms-lab/llava-next-72b --tokenizer-path lmms-lab/llavanext-qwen-tokenizer --port=30000 --host="127.0.0.1" --tp-size=4

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m sglang.launch_server --model-path lmms-lab/llama3-llava-next-8b --tokenizer-path lmms-lab/llama3-llava-next-8b-tokenizer --port=10003 --host="127.0.0.1" --tp-size=4


python serve/controller.py --host 0.0.0.0 --port 12355
python serve/gradio_web_server.py --controller-url=http://localhost:12355 --model-list-mode reload --moderate
python serve/sglang_worker.py --host 0.0.0.0 --controller http://localhost:12355 --port 40000 --worker http://localhost:40000 --sgl-endpoint http://127.0.0.1:30000 
python serve/sglang_worker.py --host 0.0.0.0 --controller http://localhost:12355 --port 3000 --worker http://localhost:3000 --sgl-endpoint http://127.0.0.1:10003
<!-- python multimodal_chat.py --sglang_port=30000 -->

pip install gradio==4.29.0
pip install httpx==0.23.3