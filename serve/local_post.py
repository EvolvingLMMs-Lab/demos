import requests

pload = {
    "text": "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<image>\nwrite an image prompt for this character without details about the surrounding.\nInclude color and details about all of these variables: age, eyes, hair, skin, expression, clothes<|im_end|>\n<|im_start|>assistant\n",
    "sampling_params": {
        "max_new_tokens": 1024,
        "temperature": 0.5,
        "top_p": 1.0,
        "stop": "<|im_end|>",
    },
    "image_data": "/tmp/gradio/fa227ae0214d48b3f05331b88e83c29b15c8d1d8/user_example_11.png",
    "stream": True,
}
url = "http://127.0.0.1:30000/generate"
response = requests.post(url, json=pload)
print(response.json())