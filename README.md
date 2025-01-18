# Triton Inference Server Playground

## Usage

- Start Triton Server by running `docker compose up -d server`
- After Triton server has started, send sample inference request by running `docker compose run --rm client`. You should see the following output:
    ```
    Request 0, batch size 1
    Image '/workspace/images/mug.jpg':
        15.349566 (504) = COFFEE MUG
        13.227467 (968) = CUP
        10.424893 (505) = COFFEEPOT
    ```

See https://github.com/triton-inference-server/server/blob/main/docs/getting_started/quickstart.md for more examples or to see how to add additional models.
