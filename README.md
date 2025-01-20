# Triton Inference Server Playground

A collection of examples for getting familiar with Triton Server.

## Usage

- Start Triton Server by running `docker compose up -d server`. Note: it may take time to start up the first time, as it will download sample models including a 0.5B LLM from Huggingface (models are cached for subsequent server container startups) - run `docker compose logs -f server` to track progress - once the server is running, you should see
    ```
    server-1  | I0120 02:08:04.566825 1 grpc_server.cc:2558] "Started GRPCInferenceService at 0.0.0.0:8001"
    server-1  | I0120 02:08:04.567038 1 http_server.cc:4725] "Started HTTPService at 0.0.0.0:8000"
    server-1  | I0120 02:08:04.614165 1 http_server.cc:358] "Started Metrics Service at 0.0.0.0:8002"
    ```
    
- After Triton server has started, send sample inference to Densenet ONNX example image classification model by running `docker compose run --rm client`. You should see the following output:
    ```
    Request 0, batch size 1
    Image '/workspace/images/mug.jpg':
        15.349566 (504) = COFFEE MUG
        13.227467 (968) = CUP
        10.424893 (505) = COFFEEPOT
    ```
- After Triton server has started, send sample inference to Qwen2.5 example text generation model by running example notebook in Jupyter container (see instructions below), or by running the following if you have curl installed on your host machine:
    ```
    curl -X POST localhost:8000/v2/models/qwen2.5/infer -d '{"inputs": [{"name":"text_input","datatype":"BYTES","shape":[1],"data":["User:Write a short story about dogs.\n\nAssistant:"]}]}'
    ```

    You should see similar output as below:
    ```
    {
        "model_name": "qwen2.5",
        "model_version": "1",
        "outputs": [
            {
                "name": "text_output",
                "datatype": "BYTES",
                "shape": [
                    1
                ],
                "data": [
                    "User:Write a short story about dogs.\n\nAssistant: Once upon a time, there was a group of friendly and loyal dogs who lived in a small town. They were called the \"Pack\" because they all had similar personalities and shared a strong bond with each other.\nThe Pack was led by a young and energetic male dog named Max, who was known for his intelligence and loyalty. He would often help the other dogs find food or water, and he always made sure to stay alert for any danger"
                ]
            }
        ]
    }
    ```

- Start Jupyter Notebook server by running `docker compose up -d jupyter`
- After Jupyter notebook has started, navigate to http://localhost:8888 and create or run existing noteboooks

See https://github.com/triton-inference-server/server/blob/main/docs/getting_started/quickstart.md for more examples or to see how to add additional models.
