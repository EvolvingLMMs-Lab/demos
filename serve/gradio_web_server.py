import argparse
import datetime
import json
import os
import time

import gradio as gr
import requests

from llava.conversation import default_conversation, conv_templates, SeparatorStyle
from llava.utils import (
    build_logger,
    server_error_msg,
    violates_moderation,
    moderation_msg,
)
import hashlib
import PIL
import shortuuid

logger = build_logger("gradio_web_server", "./logs/gradio_web_server.log")

headers = {"User-Agent": "LLaVA-NeXT Client"}

no_change_btn = gr.Button()
enable_btn = gr.Button(interactive=True)
disable_btn = gr.Button(interactive=False)

PARENT_FOLDER = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGDIR=f"{PARENT_FOLDER}/logs"
print(PARENT_FOLDER)

priority = {
    "vicuna-13b": "aaaaaaa",
    "koala-13b": "aaaaaab",
}


def get_conv_log_filename():
    t = datetime.datetime.now()
    name = os.path.join(LOGDIR, f"{t.year}-{t.month:02d}-{t.day:02d}-user_conv.json")
    return name


def get_model_list():
    ret = requests.post(args.controller_url + "/refresh_all_workers")
    assert ret.status_code == 200
    ret = requests.post(args.controller_url + "/list_models")
    models = ret.json()["models"]
    models.sort(key=lambda x: priority.get(x, x))
    logger.info(f"Models: {models}")
    return models


get_window_url_params = """
function() {
    const params = new URLSearchParams(window.location.search);
    url_params = Object.fromEntries(params);
    console.log(url_params);
    return url_params;
    }
"""


def get_new_state(template_name=None):
    if template_name is not None:
        state = conv_templates[template_name].copy()
    else:
        state = default_conversation.copy()

    state.identifier = shortuuid.uuid()
    return state


def load_demo(url_params, request: gr.Request):
    logger.info(f"load_demo. ip: {request.client.host}. params: {url_params}")

    dropdown_update = gr.Dropdown(visible=True)
    if "model" in url_params:
        model = url_params["model"]
        if model in models:
            dropdown_update = gr.Dropdown(value=model, visible=True)

    state = default_conversation.copy()
    return state, dropdown_update


def load_demo_refresh_model_list(request: gr.Request):
    logger.info(f"load_demo. ip: {request.client.host}")
    models = get_model_list()
    state = default_conversation.copy()
    dropdown_update = gr.Dropdown(
        choices=models, value=models[0] if len(models) > 0 else ""
    )
    return state, dropdown_update


def vote_last_response(state, vote_type, model_selector, request: gr.Request):
    with open(get_conv_log_filename(), "a") as fout:
        data = {
            "tstamp": round(time.time(), 4),
            "type": vote_type,
            "model": model_selector,
            "state": state.dict(),
            "ip": request.client.host,
        }
        fout.write(json.dumps(data) + "\n")


def upvote_last_response(state, model_selector, request: gr.Request):
    logger.info(f"upvote. ip: {request.client.host}")
    vote_last_response(state, "upvote", model_selector, request)
    return (gr.MultimodalTextbox(value=None, interactive=True),) + (disable_btn,) * 3


def downvote_last_response(state, model_selector, request: gr.Request):
    logger.info(f"downvote. ip: {request.client.host}")
    vote_last_response(state, "downvote", model_selector, request)
    return (gr.MultimodalTextbox(value=None, interactive=True),) + (disable_btn,) * 3


def flag_last_response(state, model_selector, request: gr.Request):
    logger.info(f"flag. ip: {request.client.host}")
    vote_last_response(state, "flag", model_selector, request)
    return (gr.MultimodalTextbox(value=None, interactive=True),) + (disable_btn,) * 3


def regenerate(state, image_process_mode, request: gr.Request):
    logger.info(f"regenerate. ip: {request.client.host}")
    state.messages[-1][-1] = None
    prev_human_msg = state.messages[-2]
    if type(prev_human_msg[1]) in (tuple, list):
        prev_human_msg[1] = (*prev_human_msg[1][:2], image_process_mode)
    state.skip_next = False
    return (
        state,
        state.to_gradio_chatbot(),
        gr.MultimodalTextbox(value=None, interactive=True),
        None,
    ) + (disable_btn,) * 5


def clear_history(request: gr.Request):
    logger.info(f"clear_history. ip: {request.client.host}")
    state = default_conversation.copy()
    return (
        state,
        state.to_gradio_chatbot(),
        gr.MultimodalTextbox(value=None, interactive=True),
        None,
    ) + (disable_btn,) * 5


def add_text(state, messages, image_process_mode, request: gr.Request):
    image = [
        PIL.Image.open(image_path).convert("RGB") for image_path in messages["files"]
    ] if len(messages["files"]) > 0 else None
    text = messages["text"]
    logger.info(f"add_text. ip: {request.client.host}. len: {len(text)}")
    if len(text) <= 0 and image is None:
        state.skip_next = True
        return (
            state,
            state.to_gradio_chatbot(),
            gr.MultimodalTextbox(value=None, interactive=True),
            None,
        ) + (no_change_btn,) * 5
    if True:
        flagged = violates_moderation(text)
        if flagged:
            logger.info(f"######################### Moderating: {text} #########################")
            logger.info(f"######################### FLAG: {flagged} #########################")
            state.skip_next = True
            mod_value = {
                "text": moderation_msg,
                "files": []
            }
            return (state, state.to_gradio_chatbot(), gr.MultimodalTextbox(value=mod_value, interactive=True), None) + (
                enable_btn, disable_btn, disable_btn, disable_btn, enable_btn
            )

    # text = text[:1536]  # Hard cut-off
    if image is not None:
        # text = text[:1200]  # Hard cut-off for images
        if "<image>" not in text:
            # text = '<Image><image></Image>' + text
            text = "<image>" * len(image) + "\n" + text
        text = (text, image, image_process_mode)
        state = default_conversation.copy()
    state.append_message(state.roles[0], text)
    state.append_message(state.roles[1], None)
    state.skip_next = False
    return (
        state,
        state.to_gradio_chatbot(),
        gr.MultimodalTextbox(value=None, interactive=False),
        None,
    ) + (disable_btn,) * 5


def http_bot(
    state, model_selector, temperature, top_p, max_new_tokens, request: gr.Request
):
    logger.info(f"http_bot. ip: {request.client.host}")
    start_tstamp = time.time()
    model_name = model_selector

    if state.skip_next:
        # This generate call is skipped due to invalid inputs
        yield (state, state.to_gradio_chatbot()) + (no_change_btn,) * 5
        return

    if len(state.messages) == state.offset + 2:
        # First round of conversation
        if "llava" in model_name.lower():
            if "llama-2" in model_name.lower():
                template_name = "llava_llama_2"
            elif "mistral" in model_name.lower() or "mixtral" in model_name.lower():
                if "orca" in model_name.lower():
                    template_name = "mistral_orca"
                elif "hermes" in model_name.lower():
                    template_name = "chatml_direct"
                else:
                    template_name = "mistral_instruct"
            elif "llava-v1.6-34b" in model_name.lower():
                template_name = "chatml_direct"
            elif (
                "llava-next-72b" in model_name.lower()
                or "llava-next-110b" in model_name.lower()
            ):
                template_name = "qwen_1_5"
            elif "llama-3" in model_name.lower():
                template_name = "llava_llama_3"
            elif "v1" in model_name.lower():
                if "mmtag" in model_name.lower():
                    template_name = "v1_mmtag"
                elif (
                    "plain" in model_name.lower()
                    and "finetune" not in model_name.lower()
                ):
                    template_name = "v1_mmtag"
                else:
                    template_name = "llava_v1"
            elif "mpt" in model_name.lower():
                template_name = "mpt"
            else:
                if "mmtag" in model_name.lower():
                    template_name = "v0_mmtag"
                elif (
                    "plain" in model_name.lower()
                    and "finetune" not in model_name.lower()
                ):
                    template_name = "v0_mmtag"
                else:
                    template_name = "llava_v0"
        elif "mistral" in model_name.lower() or "mixtral" in model_name.lower():
            if "orca" in model_name.lower():
                template_name = "mistral_orca"
            elif "hermes" in model_name.lower():
                template_name = "mistral_direct"
            else:
                template_name = "mistral_instruct"
        elif "hermes" in model_name.lower():
            template_name = "mistral_direct"
        elif "zephyr" in model_name.lower():
            template_name = "mistral_zephyr"
        elif "mpt" in model_name:
            template_name = "mpt_text"
        elif "llama-2" in model_name:
            template_name = "llama_2"
        else:
            template_name = "vicuna_v1"
        new_state = get_new_state(template_name)
        new_state.append_message(new_state.roles[0], state.messages[-2][1])
        new_state.append_message(new_state.roles[1], None)
        state = new_state

    # Query worker address
    controller_url = args.controller_url
    ret = requests.post(
        controller_url + "/get_worker_address", json={"model": model_name}
    )
    worker_addr = ret.json()["address"]
    logger.info(f"model_name: {model_name}, worker_addr: {worker_addr}")

    # No available worker
    if worker_addr == "":
        state.messages[-1][-1] = server_error_msg
        yield (
            state,
            state.to_gradio_chatbot(),
            disable_btn,
            disable_btn,
            disable_btn,
            enable_btn,
            disable_btn,
        )
        return

    # Construct prompt
    prompt = state.get_prompt()

    all_images = state.get_images(return_pil=True)
    all_image_hash = [hashlib.md5(image.tobytes()).hexdigest() for image in all_images]
    for image, hash in zip(all_images, all_image_hash):
        t = datetime.datetime.now()
        filename = os.path.join(
            LOGDIR, "serve_images", f"{t.year}-{t.month:02d}-{t.day:02d}", f"{hash}.jpg"
        )
        if not os.path.isfile(filename):
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            image.save(filename)

    # Make requests
    pload = {
        "model": model_name,
        "prompt": prompt,
        "temperature": float(temperature),
        "top_p": float(top_p),
        "max_new_tokens": min(int(max_new_tokens), 1536),
        "stop": (
            state.sep
            if state.sep_style in [SeparatorStyle.SINGLE, SeparatorStyle.MPT]
            else state.sep2
        ),
        "images": f"List of {len(state.get_images())} images: {all_image_hash}",
    }
    logger.info(f"==== request ====\n{pload}")

    pload["images"] = state.get_images()

    state.messages[-1][-1] = "▌"
    yield (state, state.to_gradio_chatbot()) + (disable_btn,) * 5

    try:
        # Stream output
        response = requests.post(
            worker_addr + "/worker_generate_stream",
            headers=headers,
            json=pload,
            stream=True,
            timeout=100,
        )
        last_print_time = time.time()
        for chunk in response.iter_lines(decode_unicode=False, delimiter=b"\0"):
            if chunk:
                data = json.loads(chunk.decode())
                if data["error_code"] == 0:
                    output = data["text"][len(prompt) :].strip()
                    state.messages[-1][-1] = output + "▌"
                    if time.time() - last_print_time > 0.05:
                        last_print_time = time.time()
                        yield (state, state.to_gradio_chatbot()) + (disable_btn,) * 5
                else:
                    output = data["text"] + f" (error_code: {data['error_code']})"
                    state.messages[-1][-1] = output
                    yield (state, state.to_gradio_chatbot()) + (
                        disable_btn,
                        disable_btn,
                        disable_btn,
                        enable_btn,
                        enable_btn,
                    )
                    return
                time.sleep(0.03)
    except requests.exceptions.RequestException as e:
        state.messages[-1][-1] = server_error_msg
        yield (state, state.to_gradio_chatbot()) + (
            disable_btn,
            disable_btn,
            disable_btn,
            enable_btn,
            enable_btn,
        )
        return

    state.messages[-1][-1] = state.messages[-1][-1][:-1]
    yield (state, state.to_gradio_chatbot()) + (enable_btn,) * 5

    finish_tstamp = time.time()
    logger.info(f"{output}")

    with open(get_conv_log_filename(), "a") as fout:
        data = {
            "tstamp": round(finish_tstamp, 4),
            "type": "chat",
            "model": model_name,
            "start": round(start_tstamp, 4),
            "finish": round(start_tstamp, 4),
            "state": state.dict(),
            "images": all_image_hash,
            "ip": request.client.host,
        }
        fout.write(json.dumps(data) + "\n")


from serve_constants import *


def print_like_dislike(x: gr.LikeData):
    print(x.index, x.value, x.liked)


def build_demo(embed_mode, cur_dir=None, concurrency_count=10):
    # textbox = gr.Textbox(
    #     show_label=False, placeholder="Enter text and press ENTER", container=False
    # )
    chatbot = gr.Chatbot(
        [],
        elem_id="chatbot",
        bubble_full_width=False,
        height=600,
        avatar_images=(
            (
                os.path.join(
                    os.path.dirname(__file__), f"{PARENT_FOLDER}/assets/user_logo.png"
                )
            ),
            (
                os.path.join(
                    os.path.dirname(__file__),
                    f"{PARENT_FOLDER}/assets/assistant_logo.png",
                )
            ),
        ),
    )

    textbox = gr.MultimodalTextbox(
        interactive=True,
        file_types=["image"],
        placeholder="Enter message or upload file...",
        show_label=False,
        max_lines=10000,
    )
    with gr.Blocks(
        theme="finlaymacklon/smooth_slate",
        title="LLaVA-NeXT: Multimodal Chat",
        css=".message-wrap.svelte-1lcyrx4>div.svelte-1lcyrx4  img {min-width: 40px}",
    ) as demo:
        state = gr.State()
        if not embed_mode:
            # gr.Markdown(title_markdown)
            gr.HTML(html_header)

        with gr.Row():
            with gr.Column(scale=1):
                model_selector = gr.Dropdown(
                    choices=models,
                    value=models[0] if len(models) > 0 else "",
                    interactive=True,
                    show_label=False,
                    container=False,
                )

                with gr.Accordion("Parameters", open=False) as parameter_row:
                    temperature = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.0,
                        step=0.1,
                        interactive=True,
                        label="Temperature",
                    )
                    top_p = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.7,
                        step=0.1,
                        interactive=True,
                        label="Top P",
                    )
                    max_output_tokens = gr.Slider(
                        minimum=0,
                        maximum=8192,
                        value=1024,
                        step=64,
                        interactive=True,
                        label="Max output tokens",
                    )

                image_process_mode = gr.Radio(
                    ["Crop", "Resize", "Pad", "Default"],
                    value="Default",
                    label="Preprocess for non-square image",
                    visible=False,
                )

                if cur_dir is None:
                    cur_dir = os.path.dirname(os.path.abspath(__file__))

                gr.Examples(
                    examples=[
                        {
                            "files": [
                                f"{PARENT_FOLDER}/assets/user_example_04.jpg",
                            ],
                            "text": "What is the latex code for this image?",
                        },
                        {
                            "files": [
                                f"{PARENT_FOLDER}/assets/user_example_11.png",
                            ],
                            "text": "write an image prompt for this character without details about the surrounding.\nInclude color and details about all of these variables: age, eyes, hair, skin, expression, clothes",
                        },
                        {
                            "files": [
                                f"{PARENT_FOLDER}/assets/user_example_06.jpg",
                            ],
                            "text": "Write the content of this table in a Notion format?",
                        },
                        {
                            "files": [
                                f"{PARENT_FOLDER}/assets/user_example_05.jpg",
                            ],
                            "text": "この猫の目の大きさは、どのような理由で他の猫と比べて特に大きく見えますか？",
                        },
                        {
                            "files": [
                                f"{PARENT_FOLDER}/assets/user_example_09.jpg",
                            ],
                            "text": "请针对于这幅画写一首中文古诗。",
                        },
                    ],
                    inputs=[textbox],
                )
                with gr.Accordion("More Examples", open=False) as more_examples_row:
                    gr.Examples(
                        examples=[
                            {
                                "files": [
                                    f"{PARENT_FOLDER}/assets/user_example_07.jpg",
                                ],
                                "text": "这个是什么猫？它在干啥？",
                            },
                            {
                                "files": [
                                    f"{PARENT_FOLDER}/assets/user_example_10.png",
                                ],
                                "text": "Here's a design for blogging website. Provide the working source code for the website using HTML, CSS and JavaScript as required.",
                            },
                        ],
                        inputs=[textbox],
                    )
            with gr.Column(scale=9):
                chatbot.render()
                textbox.render()
                with gr.Row(elem_id="buttons") as button_row:
                    clear_btn = gr.Button(value="🗑️  Clear", interactive=False)
                    upvote_btn = gr.Button(
                        value="Up-Vote", interactive=False, visible=False
                    )
                    downvote_btn = gr.Button(
                        value="Down-Vote", interactive=False, visible=False
                    )
                    flag_btn = gr.Button(value="Flag", interactive=False, visible=False)
                    regenerate_btn = gr.Button(value="🔄 Regenerate", interactive=False)
                    submit_btn = gr.Button(value="Send", variant="primary")

        if not embed_mode:
            gr.Markdown(tos_markdown)
            gr.Markdown(learn_more_markdown)
            gr.Markdown(bibtext)
        url_params = gr.JSON(visible=False)

        # Register listeners
        btn_list = [upvote_btn, downvote_btn, regenerate_btn, clear_btn, submit_btn]
        upvote_btn.click(
            upvote_last_response,
            [state, model_selector],
            [textbox, upvote_btn, downvote_btn, flag_btn],
        )
        downvote_btn.click(
            downvote_last_response,
            [state, model_selector],
            [textbox, upvote_btn, downvote_btn, flag_btn],
        )
        flag_btn.click(
            flag_last_response,
            [state, model_selector],
            [textbox, upvote_btn, downvote_btn, flag_btn],
        )

        regenerate_btn.click(
            regenerate,
            [state, image_process_mode],
            [state, chatbot, textbox] + btn_list,
        ).then(
            http_bot,
            [state, model_selector, temperature, top_p, max_output_tokens],
            [state, chatbot] + btn_list,
            concurrency_limit=concurrency_count,
        )

        clear_btn.click(
            clear_history,
            None,
            [state, chatbot, textbox] + btn_list,
            queue=False,
        )

        textbox.submit(
            add_text,
            [state, textbox, image_process_mode],
            [state, chatbot, textbox] + btn_list,
            queue=False,
        ).then(
            http_bot,
            [state, model_selector, temperature, top_p, max_output_tokens],
            [state, chatbot] + btn_list,
            concurrency_limit=concurrency_count,
        )

        chatbot.like(print_like_dislike, None, None)

        submit_btn.click(
            add_text,
            [state, textbox, image_process_mode],
            [state, chatbot, textbox] + btn_list,
        ).then(
            http_bot,
            [state, model_selector, temperature, top_p, max_output_tokens],
            [state, chatbot] + btn_list,
            concurrency_limit=concurrency_count,
        )

        if args.model_list_mode == "once":
            demo.load(
                load_demo,
                [url_params],
                [state, model_selector],
                js=get_window_url_params,
            )
        elif args.model_list_mode == "reload":
            demo.load(
                load_demo_refresh_model_list, None, [state, model_selector], queue=False
            )
        else:
            raise ValueError(f"Unknown model list mode: {args.model_list_mode}")

    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int)
    parser.add_argument("--controller-url", type=str, default="http://localhost:50030")
    parser.add_argument("--concurrency-count", type=int, default=32)
    parser.add_argument(
        "--model-list-mode", type=str, default="once", choices=["once", "reload"]
    )
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--moderate", action="store_true")
    parser.add_argument("--embed", action="store_true")
    args = parser.parse_args()
    logger.info(f"args: {args}")

    models = get_model_list()

    logger.info(args)
    demo = build_demo(args.embed, concurrency_count=args.concurrency_count)
    demo.queue(api_open=False).launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        favicon_path=f"{PARENT_FOLDER}/assets/favicon.ico",
    )
