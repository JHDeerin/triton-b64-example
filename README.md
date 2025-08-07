# Triton B64 Example

A basic example of how to pass a base64-encoded JPEG image into an [NVIDIA Triton](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/introduction/index.html) model (because I found this surprisingly annoying to figure out).

## The important bits I learned

-   Data types: your request with the b64 image should be set to `BYTES`, while the Triton server config file should set the input to `TYPE_STRING`. This was my main confusion (at first I thought that they should both be set to `STRING`/`TYPE_STRING`, but nope).
-   The request's `data` field should be an array with a single string in it, like so: `"data": ["myb64data..."]`


## How to run this

First, you'll need both [Docker](https://www.docker.com/get-started/) and [uv](https://docs.astral.sh/uv/getting-started/) installed. Then,

1.  Start the Triton server.

    ```sh
    docker build -t triton-b64-example:latest .
    docker run --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 -v "$(pwd)/triton_models:/models" triton-b64-example
    ```

    > NOTE: The Triton docker image is several GB in size, and might take a few minutes to download

2.  In a separate tab, run `uv run main.py` to make a request (converts `test.jpeg` to base64, sends a request in the right format, and prints the response).
