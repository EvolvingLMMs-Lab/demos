---
title: LLaVA
app_file: serve/gradio_web_server.py
sdk: gradio
sdk_version: 4.29.0
---
# Installation

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

<!-- pip install gradio==4.29.0 -->
pip install httpx==0.23.3
pip install flashinfer -i https://flashinfer.ai/whl/cu121/torch2.3/

# OpenAI Compatible Server
```bash
python -m sglang.launch_server --model-path lmms-lab/llama3-llava-next-8b --tokenizer-path lmms-lab/llama3-llava-next-8b-tokenizer --port=12000 --host="127.0.0.1" --tp-size=1 --chat-template llava_llama_3

python test_openai_llava.py
```

# LLaVA OneVision Test

```bash
# OV_MODEL=lmms-lab/llava-onevision-qwen2-7b-ov
python -m sglang.launch_server --model-path lmms-lab/llava-onevision-qwen2-7b-ov --tokenizer-path lmms-lab/llavanext-qwen-siglip-tokenizer --port=30000 --host="127.0.0.1" --tp-size=8 --chat-template=chatml-llava

pip install https://gradio-builds.s3.amazonaws.com/f94000877d57cf10f69d55c3aef8f6d9fd93fa7c/gradio-4.40.0-py3-none-any.whl
pip install "gradio-client @ git+https://github.com/gradio-app/gradio@f94000877d57cf10f69d55c3aef8f6d9fd93fa7c#subdirectory=client/python"

python test_openai_llava.py

python serve/gradio_web_server.py
```