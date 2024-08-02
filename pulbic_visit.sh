# curl https://llava-next-endpoint.lmms-lab.com/generate \
#     -H "Content-Type: application/json" \
#     -d '{
#         "text": "<image>\nPlease generate caption towards this image.",
#         "sampling_params": {
#         "max_new_tokens": 16,
#         "temperature": 0
#         }
#     }'

# echo " "

curl -X POST "http://127.0.0.1:10010/generate" \
    -H "Content-Type: application/json" \
    -d '{
    "text": "<image>\nPlease generate caption towards this image.",
    "image_data": "https://farm4.staticflickr.com/3175/2653711032_804ff86d81_z.jpg",
    "sampling_params": {
      "max_new_tokens": 16,
      "temperature": 0
    }
  }'
