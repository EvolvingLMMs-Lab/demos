import openai

client = openai.Client(api_key="EMPTY", base_url="https://127.0.0.1:10000/v1")
response = client.chat.completions.create(
    model="default",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://github.com/sgl-project/sglang/blob/main/test/lang/example_image.png?raw=true"
                    },
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://github.com/sgl-project/sglang/blob/main/test/lang/example_image.png?raw=true"
                    },
                },
                {
                    "type": "text",
                    "text": "Describe this image in a very short sentence.",
                },
            ],
        },
    ],
    temperature=0,
    modalities=["multi-image"],
)
print(response.choices[0].message.content)
