# Installation

curl -L --output cloudflared.deb https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb && 

sudo dpkg -i cloudflared.deb && 

sudo cloudflared service install eyJhIjoiZjMyZGYyNTNmMzVjMzA5ODBjZTMyMGM0MTUyZjZjZmEiLCJ0IjoiYTM5MzMzYWYtYTY3OC00MThlLWIyZWYtZDQzNmVkNDRhMzc0IiwicyI6Ik1qZ3laalk1TnpRdE1EQmlNeTAwWkRjM0xXRTFaR1F0WXpCbU9XSTNOelZtTmpNMyJ9

cd sglang;
pip install -e "python[srt]"

cd ..;

python -m sglang.launch_server --model-path lmms-lab/llava-next-72b --tokenizer-path lmms-lab/llavanext-qwen-tokenizer --port=30000 --host="127.0.0.1" --tp-size=8;

python multimodal_chat.py --sglang_port=30000

pip install gradio==4.29.0
pip install https==0.23.3