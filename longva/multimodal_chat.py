import gradio as gr
import os
import json
from datetime import datetime
import hashlib
import argparse
from PIL import Image

from theme_dropdown import create_theme_dropdown  # noqa: F401
from constants import (
    title_markdown,
    tos_markdown,
    learn_more_markdown,
    bibtext,
    multimodal_folder_path,
)

dropdown, js = create_theme_dropdown()

from longva import LongVA

longva = LongVA()


def generate_file_hash(file_path):
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()[:6]


def print_like_dislike(x: gr.LikeData):
    print(x.index, x.value, x.liked)


def add_message(history, message):
    for x in message["files"]:
        history.append(((x,), None))
    if message["text"] is not None:
        history.append((message["text"], None))
    return history, gr.MultimodalTextbox(value=None, interactive=False)


def longva_generate(image_path, prompt, temperature=0, max_new_tokens=8192, task_type="image"):
    if task_type == "image":
        image = Image.open(image_path).convert("RGB")
        request = {"visuals": [image], "context": prompt, "task_type": task_type}
    elif task_type == "video":
        request = {"visuals": [image_path], "context": prompt, "task_type": task_type}
        
    gen_kwargs = {"max_new_tokens": max_new_tokens, "temperature": temperature, "do_sample": False}
    response = longva.generate_until(request, gen_kwargs)
    return response


def process_image_and_prompt(image_path, prompt, temperature=0, max_new_tokens=8192):
    start_time = datetime.now()
    formated_time = start_time.strftime("%Y-%m-%d-%H-%M-%S")

    if not os.path.exists(image_path):
        return "Image is not correctly uploaded and processed. Please try again."

    print(f"Processing Visual: {image_path}")
    try:
        if (
            ".png" in image_path.lower()
            or ".jpg" in image_path.lower()
            or ".jpeg" in image_path.lower()
            or ".webp" in image_path.lower()
            or ".bmp" in image_path.lower()
            or ".gif" in image_path.lower()
        ):
            task_type = "image"
            response = longva_generate(image_path, prompt, temperature, max_new_tokens, task_type)
        elif ".mp4" in image_path.lower():
            task_type = "video"
            response = longva_generate(image_path, prompt, temperature, max_new_tokens, task_type)
        else:
            response = (
                "Image format is not supported. Please upload a valid image file."
            )
    except Exception as e:
        print(e)
        return "Image is not correctly uploaded and processed. Please try again."

    hashed_value = generate_file_hash(image_path)
    collected_json_path = os.path.join(
        multimodal_folder_path, f"{formated_time}_{hashed_value}.json"
    )
    collected_user_logs = {}
    collected_user_logs["image_path"] = image_path
    collected_user_logs["user_questions"] = prompt
    collected_user_logs["model_response"] = response

    with open(collected_json_path, "w") as f:
        f.write(json.dumps(collected_user_logs))

    print(f"################# {collected_json_path} #################")
    print(f"Visual Path: {image_path}")
    print(f"Question: {prompt}")
    print(f"Response: {response}")
    print(f"######################### END ############################")
    return response


def bot(history, temperature=0.2, max_new_tokens=8192):
    try:
        if len(history) > 2:
            history = history[-2:]
        response = process_image_and_prompt(
            history[-2][0][0], history[-1][0], temperature, max_new_tokens
        )
    except Exception as e:
        print(e)
        response = "Image is not correctly uploaded and processed. Please try again."

    try:
        history[-1][1] = ""
        for character in response:
            history[-1][1] += character
            yield history
    except Exception as e:
        print(e)
        yield history


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="LongVA-7B", help="Model name")
    parser.add_argument("--temperature", default="0", help="Temperature")
    parser.add_argument("--max_new_tokens", default="8192", help="Max new tokens")
    args = parser.parse_args()
    with gr.Blocks(
        theme="finlaymacklon/smooth_slate",
        title="LongVA Multimodal Chat from LMMs-Lab",
        css=".message-wrap.svelte-1lcyrx4>div.svelte-1lcyrx4  img {min-width: 50px}",
    ) as demo:
        gr.Markdown(title_markdown)

        # model_selector = gr.Dropdown(
        #     choices=models,
        #     value=models[0] if len(models) > 0 else "",
        #     interactive=True,
        #     show_label=False,
        #     container=False)

        chatbot = gr.Chatbot(
            [],
            label=f"Model: {args.model_name}",
            elem_id="chatbot",
            bubble_full_width=False,
            height=600,
            avatar_images=(
                (
                    os.path.join(
                        os.path.dirname(__file__),
                        "/mnt/bn/vl-research/workspace/boli01/projects/demos/assets/user_logo.png",
                    )
                ),
                (
                    os.path.join(
                        os.path.dirname(__file__),
                        "/mnt/bn/vl-research/workspace/boli01/projects/demos/assets/assistant_logo.png",
                    )
                ),
            ),
        )

        chat_input = gr.MultimodalTextbox(
            interactive=True,
            file_types=["image", "video"],
            placeholder="Enter message or upload file...",
            show_label=False,
            max_lines=10000,
        )

        chat_msg = chat_input.submit(
            add_message, [chatbot, chat_input], [chatbot, chat_input]
        )
        bot_msg = chat_msg.then(
            bot, inputs=[chatbot], outputs=[chatbot], api_name="bot_response"
        )
        bot_msg.then(lambda: gr.MultimodalTextbox(interactive=True), None, [chat_input])

        chatbot.like(print_like_dislike, None, None)

        with gr.Row():
            gr.ClearButton(chatbot, chat_input, chat_msg, bot_msg)
            submit_btn = gr.Button("Send", chat_msg)
            submit_btn.click(
                add_message, [chatbot, chat_input], [chatbot, chat_input]
            ).then(bot, chatbot, chatbot, api_name="bot_response").then(
                lambda: gr.MultimodalTextbox(interactive=True), None, [chat_input]
            )

        gr.Examples(
            examples=[
                {
                    "files": [
                        "/mnt/bn/vl-research/workspace/boli01/projects/demos/assets/user_example_01.jpg",
                    ],
                    "text": "Explain this diagram.",
                },
                {
                    "files": [
                        "/mnt/bn/vl-research/workspace/boli01/projects/demos/assets/user_example_03.jpg",
                    ],
                    "text": "What characters are used in this captcha? Write them sequentially",
                },
                {
                    "files": [
                        "/mnt/bn/vl-research/workspace/boli01/projects/demos/assets/user_example_04.jpg",
                    ],
                    "text": "What is the latex code for this image?",
                },
                {
                    "files": [
                        "/mnt/bn/vl-research/workspace/boli01/projects/demos/assets/user_example_06.jpg",
                    ],
                    "text": "Write the content of this table in a Notion format?",
                },
                {
                    "files": [
                        "/mnt/bn/vl-research/workspace/boli01/projects/demos/assets/user_example_07.jpg",
                    ],
                    "text": "这个是什么猫？它在干啥？",
                },
                {
                    "files": [
                        "/mnt/bn/vl-research/workspace/boli01/projects/demos/assets/user_example_05.jpg",
                    ],
                    "text": "この猫の目の大きさは、どのような理由で他の猫と比べて特に大きく見えますか？",
                },
                {
                    "files": [
                        "/mnt/bn/vl-research/workspace/boli01/projects/demos/assets/user_example_09.jpg",
                    ],
                    "text": "请针对于这幅画写一首中文古诗。",
                },
            ],
            inputs=[chat_input],
        )
        gr.Markdown(bibtext)
        gr.Markdown(tos_markdown)
        gr.Markdown(learn_more_markdown)

    demo.queue(max_size=128)
    demo.launch(max_threads=8)
