#start triton inference server

docker run --gpus=1 -it --rm --shm-size=8g -p 8005:8000  -v ./triton_inference_server/:/model_dir  nvcr.io/nvidia/tritonserver:22.10-py3 tritonserver \
                    --model-repository=/model_dir --strict-model-config=false --model-control-mode=poll \--repository-poll-secs=10 --backend-config=tensorflow,version=2 --log-verbose=1


