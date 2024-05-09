"""
Usage:
python3 -m sglang.launch_server --model-path liuhaotian/llava-v1.5-7b --tokenizer-path llava-hf/llava-1.5-7b-hf --port 30000
python3 test_httpserver_llava.py

Output:
The image features a man standing on the back of a yellow taxi cab, holding
"""

import argparse
import asyncio
import json
import time

import aiohttp
import requests

from llava.conversation import (
    default_conversation,
    conv_templates,
    SeparatorStyle,
    conv_llava_llama_3,
    conv_qwen,
)


async def send_request(url, data, delay=0):
    await asyncio.sleep(delay)
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=data) as resp:
            output = await resp.json()
    return output


async def test_concurrent(args):
    url = f"{args.host}:{args.port}"

    response = []
    for i in range(1):
        response.append(
            send_request(
                url + "/generate",
                {
                    "text": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful language and vision assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language.<|eot_id|><|start_header_id|><|start_header_id|>user<|end_header_id|>\n\n<image>\nPlease generate caption towards this image.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
                    "image_data": "/mnt/bn/vl-research/workspace/boli01/projects/demos/assets/user_example_03.jpg",
                    "sampling_params": {
                        "temperature": 0,
                        "max_new_tokens": 1024,
                    },
                },
            )
        )

    rets = await asyncio.gather(*response)
    for ret in rets:
        print(ret["text"])


from PIL import Image
import base64


def test_streaming(args):
    url = f"{args.host}:{args.port}"
    pload = {
        "text": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful language and vision assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language.<|eot_id|><|start_header_id|><|start_header_id|>user<|end_header_id|>\n\n<image>\nこの猫の目の大きさは、どのような理由で他の猫と比べて特に大きく見えますか？<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
        "sampling_params": {
            "max_new_tokens": 1024,
            "temperature": 0.5,
            "top_p": 1.0,
            "presence_penalty": 2,
            "frequency_penalty": 2,
            "stop": "<|eot_id|>",
        },
        "image_data": "/tmp/gradio/b7b8f04e74cda37d9a574c1a2098d7e4ee97b212/user_example_05.jpg",
        "stream": True,
    }
    response = requests.post(
        url + "/generate",
        json=pload,
        stream=True,
    )
    # response = requests.post(
    #     url + "/generate",
    #     json={
    #         "text": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful language and vision assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language.<|eot_id|><|start_header_id|><|start_header_id|>user<|end_header_id|>\n\n<image>\nPlease generate caption towards this image.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
    #         "image_data": "/mnt/bn/vl-research/workspace/boli01/projects/demos/assets/user_example_03.jpg",
    #         "sampling_params": {
    #             "temperature": 0,
    #             "max_new_tokens": 1024,
    #         },
    #         "stream": True,
    #     },
    #     stream=True,
    # )

    prev = 0
    for chunk in response.iter_lines(decode_unicode=False):
        chunk = chunk.decode("utf-8")
        if chunk and chunk.startswith("data:"):
            if chunk == "data: [DONE]":
                break
            data = json.loads(chunk[5:].strip("\n"))
            output = data["text"].strip()
            print(output[prev:], end="", flush=True)
            prev = len(output)
    print("")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="http://127.0.0.1")
    parser.add_argument("--port", type=int, default=30000)
    args = parser.parse_args()

    # asyncio.run(test_concurrent(args))

    test_streaming(args)
