---
title: LLaVA
app_file: serve/gradio_web_server.py
sdk: gradio
sdk_version: 4.29.0
---
# Installation

```bash
prepare_demo;
build_llava;

cd sglang_codebase;
pip install -e "python[srt]"
pip install flashinfer -i https://flashinfer.ai/whl/cu121/torch2.3/;

cd ..;

pip install -U gradio;
pip install httpx==0.23.3;

python3 -m sglang.launch_server --model-path MMSFT/sft_llava_qwen_2_1M_rewrite_resupply --tokenizer-path lmms-lab/llavanext-qwen-siglip-tokenizer --port=30000 --host=127.0.0.1 --chat-template=chatml-llava;

# after installing sglang, you can then run the following to start the controller and the gradio web server, and the sglang worker to listen for requests

python serve/controller.py --host 0.0.0.0 --port 12355

python serve/gradio_web_server.py --controller-url=http://localhost:12355 --model-list-mode reload --moderate --share

python serve/sglang_worker.py --host 0.0.0.0 --controller http://localhost:12355 --port 3005 --worker http://localhost:3005 --sgl-endpoint http://127.0.0.1:30000
```

<!-- python multimodal_chat.py --sglang_port=30000 -->

<!-- pip install gradio==4.29.0 -->

# OpenAI Compatible Server
```bash
python -m sglang.launch_server --model-path lmms-lab/llama3-llava-next-8b --tokenizer-path lmms-lab/llama3-llava-next-8b-tokenizer --port=12000 --host="127.0.0.1" --tp-size=1 --chat-template llava_llama_3

python test_openai_llava.py
```

# LLaVA OneVision Test

```bash
# OV_MODEL=lmms-lab/llava-onevision-qwen2-7b-ov
python -m sglang.launch_server --model-path lmms-lab/llava-onevision-qwen2-7b-ov --tokenizer-path lmms-lab/llavanext-qwen-siglip-tokenizer --port=30000 --host="127.0.0.1" --tp-size=8 --chat-template=chatml-llava

pip install "gradio-client @ git+https://github.com/gradio-app/gradio@f94000877d57cf10f69d55c3aef8f6d9fd93fa7c#subdirectory=client/python"
pip install https://gradio-builds.s3.amazonaws.com/f94000877d57cf10f69d55c3aef8f6d9fd93fa7c/gradio-4.40.0-py3-none-any.whl

python test_openai_llava.py

python serve/gradio_web_server.py
```