FROM nvcr.io/nvidia/tritonserver:24.12-py3

COPY requirements.txt .

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

COPY ./model_repository/ /models/

COPY fetch_models.sh .

RUN ./fetch_models.sh

CMD ["tritonserver", "--model-repository", "/models"]