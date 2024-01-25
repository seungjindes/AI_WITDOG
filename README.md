bark2text


docker run --gpus=1 -it --rm --shm-size=8g -p 8005:8000  -v ./triton/:/model_dir  nvcr.io/nvidia/tritonserver:23.12-py3 tritonserver --model-repository=/model_dir --strict-model-config=false --model-control-mode=poll --repository-poll-secs=10 --backend-config=tensorflow,version=2 --log-verbose=1
