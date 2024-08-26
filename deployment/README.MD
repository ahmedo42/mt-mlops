## Usage

> All commands below are run from this directory.

### Building your model image

    Build the container locally and give it a tag

    ```bash
    docker build -t dyu-fr-inference:latest .
    ```

### Local testing

1. After building the Kserve predictor image that contains your model, spin it up to test your model inference

    ```bash
    docker compose up -d
    docker compose logs
    ```

2. Finally, send a payload to your model to test its response. To do this, use the `curl` cmmand to send a `POST` request with an example JSON payload.

    > Run this from another terminal (remember to navigate to this folder first)

    Linux/Mac Bash/zsh

    ```bash
    curl -X POST http://localhost:8080/v2/models/model/infer -H 'Content-Type: application/json' -d @./input.json
    ```