import argparse
import base64
import datetime
import json
import os
import openai
import time
import sys
import shutil

import gradio as gr
import requests

from decord import VideoReader, cpu
import io
from PIL import Image

from llava.conversation import (
    default_conversation,
    conv_templates,
    SeparatorStyle,
    conv_llava_llama_3,
    conv_qwen,
)
from llava.utils import (
    build_logger,
    server_error_msg,
    violates_moderation,
    moderation_msg,
)
import hashlib
import PIL
import shortuuid
from black_magic_utils import process_images


cur_file_path = os.path.dirname(os.path.abspath(__file__))
os.makedirs(f"{cur_file_path}/logs", exist_ok=True)
logger = build_logger(
    "gradio_web_server", f"{cur_file_path}/logs/gradio_web_server.log"
)

headers = {"User-Agent": "LLaVA-NeXT Client"}

no_change_btn = gr.Button()
enable_btn = gr.Button(interactive=True)
disable_btn = gr.Button(interactive=False)

PARENT_FOLDER = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGDIR = f"{PARENT_FOLDER}/logs"
print(PARENT_FOLDER)

priority = {
    "llama-3-llava-next-8b": "aaaaaa",
    "llava-next-72b": "aaaaaab",
}


def get_conv_log_filename():
    t = datetime.datetime.now()
    name = os.path.join(LOGDIR, f"{t.year}-{t.month:02d}-{t.day:02d}-user_conv.json")
    return name


def get_request_limit_by_ip_filename():
    t = datetime.datetime.now()
    name = os.path.join(
        LOGDIR, f"{t.year}-{t.month:02d}-{t.day:02d}-request_limit_by_ip.json"
    )
    return name


def get_model_list():
    ret = requests.post(args.controller_url + "/refresh_all_workers")
    assert ret.status_code == 200
    ret = requests.post(args.controller_url + "/list_models")
    models = ret.json()["models"]
    models.sort(key=lambda x: priority.get(x, x))
    logger.info(f"Models: {models}")
    return models


def get_worker_status(model_name):
    ret = requests.post(
        args.controller_url + "/get_worker_status", json={"model": model_name}
    )
    return ret.json()


get_window_url_params = """
function() {
    const params = new URLSearchParams(window.location.search);
    url_params = Object.fromEntries(params);
    console.log(url_params);
    return url_params;
    }
"""

from transformers import AutoTokenizer


def get_new_state(template_name=None):
    if "llama_3" in template_name:
        state = conv_llava_llama_3.copy()
        if state.tokenizer is None:
            state.tokenizer = AutoTokenizer.from_pretrained(
                "meta-llama/Meta-Llama-3-8B-Instruct"
            )
            state.tokenizer_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    elif "qwen" in template_name:
        state = conv_qwen.copy()
    elif template_name is not None:
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
    return (gr.MultimodalTextbox(value=None, interactive=True),) + (disable_btn,) * 5


def downvote_last_response(state, model_selector, request: gr.Request):
    logger.info(f"downvote. ip: {request.client.host}")
    vote_last_response(state, "downvote", model_selector, request)
    return (gr.MultimodalTextbox(value=None, interactive=True),) + (disable_btn,) * 5


def flag_last_response(state, model_selector, request: gr.Request):
    logger.info(f"flag. ip: {request.client.host}")
    vote_last_response(state, "flag", model_selector, request)
    return (gr.MultimodalTextbox(value=None, interactive=True),) + (disable_btn,) * 5


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


RPM_LIMIT = 300


def add_text(video_input, state, messages, image_process_mode, request: gr.Request):
    image = (
        [image_path for image_path in messages["files"]]
        if len(messages["files"]) > 0
        else None
    )
    text = messages["text"]
    logger.info(f"add_text. ip: {request.client.host}. len: {len(text)}")

    if state is None:
        state = default_conversation.copy()

    # Moderation
    if len(text) <= 0 and image is None:
        state.skip_next = True
        return (
            state,
            state.to_gradio_chatbot(),
            gr.MultimodalTextbox(value=None, interactive=True),
            None,
        ) + (no_change_btn,) * 5

    if args.moderate:
        flagged = violates_moderation(text)
        if flagged:
            logger.info(
                f"######################### Moderating: {text} #########################"
            )
            logger.info(
                f"######################### FLAG: {flagged} #########################"
            )
            state.skip_next = True
            mod_value = {"text": moderation_msg, "files": []}
            return (
                state,
                state.to_gradio_chatbot(),
                gr.MultimodalTextbox(value=mod_value, interactive=True),
                None,
            ) + (disable_btn,) * 5

    ################ Multi Image Check #########################
    # if image is not None and type(image) is list and len(image) > 1:
    #     if "BEAST_MODE:" not in text:
    #         logger.info(f"######################### NOT READY FOR MULTIPLE IMAGES #########################")
    #         state.skip_next = True
    #         mod_value = {
    #             "text": "OXIXM TLDV - Sorry, I'm trained on single images and currently not ready for multiple images yet. - OXIXM TLDV",
    #             "files": []
    #         }
    #         return (state, state.to_gradio_chatbot(), gr.MultimodalTextbox(value=mod_value, interactive=True), None) + (disable_btn,) * 5
    #     else:
    #         text = text.replace("BEAST_MODE:", "").strip()

    # Request limit by IP
    rpm_data = {}
    current_user_request_by_ip_file = get_request_limit_by_ip_filename()
    if not os.path.isfile(current_user_request_by_ip_file):
        with open(current_user_request_by_ip_file, "w") as fout:
            json.dump({}, fout)

    with open(current_user_request_by_ip_file, "r") as fout:
        rpm_data = json.load(fout)

    if request.client.host not in rpm_data:
        rpm_data[request.client.host] = 0

    rpm_data[request.client.host] += 1
    with open(current_user_request_by_ip_file, "w") as fout:
        json.dump(rpm_data, fout)

    if rpm_data[request.client.host] >= RPM_LIMIT:
        state.skip_next = True
        logger.info(
            f"######################### IP: {request.client.host} #########################"
        )
        logger.info(
            f"######################### Requests: {rpm_data[request.client.host]} #########################"
        )
        ret_value = {
            "text": "Sorry, you have reached the request limit. Please try again tomorrow.",
            "files": [],
        }
        return (
            state,
            state.to_gradio_chatbot(),
            gr.MultimodalTextbox(value=ret_value, interactive=True),
            None,
        ) + (disable_btn,) * 5

    # text = text[:1536]  # Hard cut-off
    if image is not None:
        # text = text[:1200]  # Hard cut-off for images
        # if "<image>" not in text:
        #     # text = '<Image><image></Image>' + text
        #     text = "<image>" * len(image) + "\n" + text
        # do not use it since openai compatible API do not need <image>
        text = (text, image, image_process_mode)
        # state = default_conversation.copy()
    elif video_input is not None:
        text = (text, [video_input], image_process_mode)
        # state = default_conversation.copy()

    state.append_message(state.roles[0], text)
    state.append_message(state.roles[1], None)
    state.skip_next = False
    return (
        state,
        state.to_gradio_chatbot(),
        gr.MultimodalTextbox(value=None, interactive=False),
        None,
    ) + (disable_btn,) * 5


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def extract_video_frames(video_path, frame_count=32):
    video_frames = []
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frames = len(vr)
    frame_interval = max(total_frames // frame_count, 1)

    for i in range(0, total_frames, frame_interval):
        frame = vr[i].asnumpy()
        frame_image = Image.fromarray(frame)
        buffered = io.BytesIO()
        frame_image.save(buffered, format="JPEG")
        frame_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        video_frames.append(frame_base64)
        if len(video_frames) >= frame_count:
            break

    # Ensure at least one frame is returned if total frames are less than required
    if len(video_frames) < frame_count and total_frames > 0:
        for i in range(total_frames):
            frame = vr[i].asnumpy()
            frame_image = Image.fromarray(frame)
            buffered = io.BytesIO()
            frame_image.save(buffered, format="JPEG")
            frame_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
            video_frames.append(frame_base64)
            if len(video_frames) >= frame_count:
                break

    return video_frames


def convert_state_to_openai_messages(state):
    messages = []

    # Add system message if present
    # if state.system:
    #     messages.append({
    #         "role": "system",
    #         "content": state.system.replace('<|im_start|>system\n', '')
    #     })

    # Convert user and assistant messages
    for message in state.messages:
        role = message[0].replace("<|im_start|>", "")
        content = message[1]

        if isinstance(content, tuple) and len(content) == 3:
            text, image_paths, _ = content
            message_content = []
            for image_path in image_paths:
                if state.is_image_file(image_path):
                    message_content.append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{encode_image(image_path)}"
                            },
                        }
                    )
                elif state.is_video_file(image_path):
                    video_frames = extract_video_frames(image_path, frame_count=32)
                    for frame in video_frames:
                        message_content.append(
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{frame}"},
                            }
                        )

            message_content.append(
                {
                    "type": "text",
                    "text": text.replace("<image>", "").strip("\n").strip(),
                }
            )
        else:
            message_content = content

        if role == "assistant" and message_content == "▌":
            continue
        else:
            messages.append({"role": role, "content": message_content})

    return messages


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
            if "llava-v1.6-34b" in model_name.lower():
                template_name = "chatml_direct"
            elif (
                "llava-next-72b" in model_name.lower()
                or "llava-next-110b" in model_name.lower()
                or "qwen" in model_name.lower()
            ):
                template_name = "qwen_1_5"
            elif "llama3" in model_name.lower():
                template_name = "llava_llama_3"

        new_state = get_new_state(template_name)
        new_state.append_message(new_state.roles[0], state.messages[-2][1])
        new_state.append_message(new_state.roles[1], None)
        state = new_state

        logger.info(
            f"============================= {template_name} ============================="
        )

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
            enable_btn,
        )
        return

    # Construct prompt
    prompt = state.get_prompt()

    all_images_path = state.get_images(return_pil=False, return_path=True)
    all_images = []
    all_image_hash = []
    for image_path in all_images_path:
        if state.is_image_file(image_path):
            with open(image_path, "rb") as image_file:
                image_data = image_file.read()
                image_hash = hashlib.md5(image_data).hexdigest()
                all_image_hash.append(image_hash)
                image = PIL.Image.open(image_path).convert("RGB")
                all_images.append(image)
                t = datetime.datetime.now()
                filename = os.path.join(
                    LOGDIR,
                    "serve_images",
                    f"{t.year}-{t.month:02d}-{t.day:02d}",
                    f"{image_hash}.jpg",
                )
                if not os.path.isfile(filename):
                    os.makedirs(os.path.dirname(filename), exist_ok=True)
                    image.save(filename)

        elif state.is_video_file(image_path):
            all_images.append(image_path)
            base_filename = os.path.basename(image_path)
            filename = os.path.join(LOGDIR, "serve_images", base_filename)
            if not os.path.isfile(filename):
                shutil.copy(image_path, filename)

    stop_style = "###"
    if state.sep_style == SeparatorStyle.LLAMA_3:
        stop_style = "<|eot_id|>"
    elif state.sep_style == SeparatorStyle.QWEN:
        stop_style = "<|im_end|>"
    elif state.sep_style in [SeparatorStyle.SINGLE, SeparatorStyle.CHATML]:
        stop_style = state.sep
    else:
        raise ValueError(f"Unknown separator style: {state.sep_style}")

    # pload = {
    #     "text": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful language and vision assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language.<|eot_id|><|start_header_id|><|start_header_id|>user<|end_header_id|>\n\n<image>\nPlease generate caption towards this image.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
    #     "sampling_params": {
    #         "temperature": 0,
    #         "max_new_tokens": 1024,
    #     },
    #     "image_data": "/mnt/bn/vl-research/workspace/boli01/projects/demos/assets/user_example_03.jpg",
    #     "stream": True,
    # }
    # Make requests
    # if len(all_images_path) > 1:
    #     image_data = process_images(all_images_path, size=1008)
    #     t = datetime.datetime.now()
    #     hash = hashlib.md5(image_data.tobytes()).hexdigest()
    #     filename = os.path.join(
    #         LOGDIR, "serve_images", f"{t.year}-{t.month:02d}-{t.day:02d}", f"{hash}.jpg"
    #     )
    #     if not os.path.isdir(os.path.dirname(filename)):
    #         os.makedirs(os.path.dirname(filename), exist_ok=True)
    #     image_data.save(filename)
    #     image_data = filename
    # else:
    #     image_data = all_images_path[0]

    pload = {
        "text": prompt,
        "sampling_params": {
            "max_new_tokens": min(int(max_new_tokens), 8192),
            "temperature": float(temperature),
            "top_p": float(top_p),
            # "presence_penalty": 2,
            # "frequency_penalty": 2,
            "stop": stop_style,
        },
        "image_data": None,
        "stream": True,
    }
    # logger.info(f"===================================== request =====================================\n{pload}")

    # pload["images"] = state.get_images()

    state.messages[-1][-1] = "▌"
    yield (state, state.to_gradio_chatbot()) + (disable_btn,) * 5

    try:
        query_endpoint = requests.post(
            worker_addr + "/worker_get_status",
            headers=headers,
            json=pload,
            timeout=100,
        ).json()["sgl_endpoint"]

        client = openai.Client(
            api_key="EMPTY", base_url=f"{query_endpoint}/v1"
        )  # http://127.0.0.1:30000/v1
        openai_compatible_msg = convert_state_to_openai_messages(state)
        response = client.chat.completions.create(
            model="default",
            messages=openai_compatible_msg,
            stream=True,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_new_tokens,
        )
        # Stream output
        # response = requests.post(
        #     query_endpoint + "/generate",
        #     json=pload,
        #     stream=True,
        #     timeout=100,
        # )
        last_print_time = time.time()
        response_text = ""
        for chunk in response:
            if chunk.choices[0].delta.content is not None:
                content = chunk.choices[0].delta.content
                response_text += content
                state.messages[-1][-1] = response_text + "▌"
                yield (state, state.to_gradio_chatbot()) + (disable_btn,) * 5
                # sys.stdout.write(content)
                # sys.stdout.flush()
                time.sleep(0.03)

        if response_text:
            state.messages[-1][-1] = response_text
            yield (state, state.to_gradio_chatbot()) + (enable_btn,) * 5
        else:
            state.messages[-1][-1] = server_error_msg
            yield (state, state.to_gradio_chatbot()) + (
                disable_btn,
                disable_btn,
                disable_btn,
                enable_btn,
                enable_btn,
            )
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

    finish_tstamp = time.time()
    logger.info(f"{response_text}")

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
        height=660,
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
        file_types=["image", "video"],
        # file_count="multiple",
        placeholder="Enter message or upload file...",
        show_label=False,
        max_lines=10000,
    )
    with gr.Blocks(
        theme="finlaymacklon/smooth_slate",
        title="LLaVA OneVision",
        css=".message-wrap.svelte-1lcyrx4>div.svelte-1lcyrx4  img {min-width: 40px}",
    ) as demo:
        state = gr.State()
        if not embed_mode:
            # gr.Markdown(title_markdown)
            gr.HTML(html_header)

        with gr.Row():
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
                    value=0.3,
                    step=0.1,
                    interactive=True,
                    label="Temperature",
                )
                top_p = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=1,
                    step=0.1,
                    interactive=True,
                    label="Top P",
                )
                max_output_tokens = gr.Slider(
                    minimum=0,
                    maximum=8192,
                    value=768,
                    step=256,
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

            # with gr.Row(scale=9):
            flag_btn = gr.Button(value="Flag", interactive=False, visible=False)

        # with gr.Row():
        chatbot.render()
        textbox.render()
        # flag_btn = gr.Button(value="Flag", interactive=False, visible=False)
        with gr.Row(elem_id="buttons") as button_row:
            clear_btn = gr.Button(value="🧹 Clear", interactive=False)
            upvote_btn = gr.Button(
                value="🤞 Upvote",
                interactive=False,
            )
            downvote_btn = gr.Button(
                value="💔 Downvote",
                interactive=False,
            )
            regenerate_btn = gr.Button(value="🔄 Regenerate", interactive=False)
            submit_btn = gr.Button(value="🌟 Send", variant="primary")

        with gr.Column(scale=10):
            video = gr.Video(label="Input Video", visible=False)

            gr.Examples(
                examples_per_page=6,
                examples=[
                    [
                        {
                            "files": [
                                f"{PARENT_FOLDER}/assets/user_example_06.jpg",
                            ],
                            "text": "Write the content of this table in a Notion format?",
                        },
                    ],
                    [
                        {
                            "files": [
                                f"{PARENT_FOLDER}/assets/otter_books.jpg",
                            ],
                            "text": "Why these two animals are reading books?",
                        },
                    ],
                    [
                        {
                            "files": [
                                f"{PARENT_FOLDER}/assets/user_example_07.jpg",
                            ],
                            "text": "那要我问问你，你这个是什么🐱？",
                        },
                    ],
                    [
                        {
                            "files": [
                                f"{PARENT_FOLDER}/assets/user_example_05.jpg",
                            ],
                            "text": "この猫の目の大きさは、どのような理由で他の猫と比べて特に大きく見えますか？",
                        },
                    ],
                    [
                        {
                            "files": [
                                f"{PARENT_FOLDER}/assets/172197131626056_P7966202.png",
                            ],
                            "text": "Why this image funny?",
                        },
                    ],
                    [
                        {
                            "files": [
                                f"{PARENT_FOLDER}/assets/user_example_11.png",
                            ],
                            "text": "write an image prompt for this character without details about the surrounding.\nInclude color and details about all of these variables: age, eyes, hair, skin, expression, clothes",
                        },
                    ],
                    [
                        {
                            "files": [
                                f"{PARENT_FOLDER}/assets/user_example_10.png",
                            ],
                            "text": "Here's a design for blogging website.\nProvide the working source code for the website using HTML, CSS and JavaScript as required.",
                        },
                    ],
                ],
                inputs=[textbox],
                label="Image",
            )
                        
            gr.Examples(
                label="Video",
                examples=[
                    [
                        {
                            "files": [
                                f"{PARENT_FOLDER}/assets/cpRSZOcJrs029ixB.mp4",
                            ],
                            "text": "What's the unusual part of this video?",
                        },
                    ],
                    [
                        {
                            "files": [
                                f"{PARENT_FOLDER}/assets/aquarium-nyc.mp4",
                            ],
                            "text": "What's the creative part of this video?",
                        },
                    ],
                    [
                        {
                            "files": [
                                f"{PARENT_FOLDER}/assets/cpRSZOcJrs029ixB.mp4",
                                f"{PARENT_FOLDER}/assets/aquarium-nyc.mp4",
                            ],
                            "text": "What is the difference between these two videos?",
                        },
                    ],
                ],
                inputs=[textbox],
            )

            gr.Examples(
                examples_per_page=4,
                inputs=[textbox],
                label="Multi-Image",
                examples=[
                    [
                        {
                            "files": [
                                f"{PARENT_FOLDER}/assets/mistral-large-2407-multiple.png",
                                f"{PARENT_FOLDER}/assets/mistral-large-2407-language-diversity.png",
                            ],
                            "text": "Conclude based on above news.",
                        }
                    ],
                    [
                        {
                            "files": [
                                f"{PARENT_FOLDER}/assets/shub.jpg",
                                f"{PARENT_FOLDER}/assets/shuc.jpg",
                                f"{PARENT_FOLDER}/assets/shud.jpg",
                            ],
                            "text": "what is fun about the images?",
                        },
                    ],
                    [
                        {
                            "files": [
                                f"{PARENT_FOLDER}/assets/iphone-15-price-1024x576.jpg",
                                f"{PARENT_FOLDER}/assets/dynamic-island-1024x576.jpg",
                                f"{PARENT_FOLDER}/assets/iphone-15-colors-1024x576.jpg",
                                f"{PARENT_FOLDER}/assets/Iphone-15-Usb-c-charger-1024x576.jpg",
                                f"{PARENT_FOLDER}/assets/A-17-processors-1024x576.jpg",
                            ],
                            "text": "The images are the PPT of iPhone 15 review. can you summarize the main information?",
                        }
                    ],
                    [
                        {
                            "files": [
                                f"{PARENT_FOLDER}/assets/fangao3.jpeg",
                                f"{PARENT_FOLDER}/assets/fangao2.jpeg",
                                f"{PARENT_FOLDER}/assets/fangao1.jpeg",
                            ],
                            "text": "Do you kown who draw these paintings?",
                        },
                    ],
                    [
                        {
                            "files": [
                                f"{PARENT_FOLDER}/assets/oprah-winfrey-resume.png",
                                f"{PARENT_FOLDER}/assets/steve-jobs-resume.jpg",
                            ],
                            "text": "Hi, there are two candidates, can you provide a brief description for each of them for me?",
                        },
                    ],
                    [
                        {
                            "files": [
                                f"{PARENT_FOLDER}/assets/original_bench.jpeg",
                                f"{PARENT_FOLDER}/assets/changed_bench.jpeg",
                            ],
                            "text": "How to edit image1 to make it look like image2?",
                        },
                    ],
                    [
                        {
                            "files": [
                                f"{PARENT_FOLDER}/assets/twitter2.jpeg",
                                f"{PARENT_FOLDER}/assets/twitter3.jpeg",
                                f"{PARENT_FOLDER}/assets/twitter4.jpeg",
                            ],
                            "text": "Please write a twitter blog post with the images.",
                        }
                    ],
                ],
            )

            # with gr.Accordion("More Examples", open=False) as more_examples_row:
            #     gr.Examples(
            #         examples=[
            #             {
            #                 "files": [
            #                     f"{PARENT_FOLDER}/assets/otter_books.jpg",
            #                 ],
            #                 "text": "Why these two animals are reading books?",
            #             },
            #             {
            #                 "files": [
            #                     f"{PARENT_FOLDER}/assets/user_example_09.jpg",
            #                 ],
            #                 "text": "请针对于这幅画写一首中文古诗。",
            #             },
            #             {
            #                 "files": [
            #                     f"{PARENT_FOLDER}/assets/white_cat_smile.jpg",
            #                 ],
            #                 "text": "Why this cat smile?",
            #             },
            #             {
            #                 "files": [
            #                     f"{PARENT_FOLDER}/assets/user_example_07.jpg",
            #                 ],
            #                 "text": "这个是什么猫？",
            #             },
            #         ],
            #         inputs=[textbox],
            #     )

        if not embed_mode:
            gr.Markdown(tos_markdown)
            gr.Markdown(learn_more_markdown)
            gr.Markdown(bibtext)
        url_params = gr.JSON(visible=False)

        # Register listeners
        btn_list = [clear_btn, upvote_btn, downvote_btn, regenerate_btn, submit_btn]
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
            [video, state, textbox, image_process_mode],
            [state, chatbot, textbox] + btn_list,
        ).then(
            http_bot,
            [state, model_selector, temperature, top_p, max_output_tokens],
            [state, chatbot] + btn_list,
            concurrency_limit=concurrency_count,
        ).then(
            lambda: gr.MultimodalTextbox(interactive=True), None, [textbox]
        ).then(
            lambda: video, None, [video]
        )

        chatbot.like(print_like_dislike, None, None)

        submit_btn.click(
            add_text,
            [video, state, textbox, image_process_mode],
            [state, chatbot, textbox] + btn_list,
        ).then(
            http_bot,
            [state, model_selector, temperature, top_p, max_output_tokens],
            [state, chatbot] + btn_list,
            concurrency_limit=concurrency_count,
        ).then(
            lambda: gr.MultimodalTextbox(interactive=True), None, [textbox]
        ).then(
            lambda: video, None, [video]
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
    parser.add_argument("--port", type=int, default=7880)
    parser.add_argument("--controller-url", type=str, default="http://localhost:12355")
    parser.add_argument("--concurrency-count", type=int, default=32)
    parser.add_argument(
        "--model-list-mode", type=str, default="reload", choices=["once", "reload"]
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
        # server_name=args.host,
        share=True,
        server_port=args.port,
        favicon_path=f"{PARENT_FOLDER}/assets/favicon.ico",
    )
