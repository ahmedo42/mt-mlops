import re

from kserve import (
    InferOutput,
    InferRequest,
    InferResponse,
    Model,
    ModelServer,
)
from kserve.utils.utils import generate_uuid
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

HF_USERNAME = "" # replace with HuggingFace Hub username.
HF_REPO_NAME = "" # replace with HuggingFace Hub repo name where the model weights are stored.
MODEL_REPO = f"{HF_USERNAME}/{HF_REPO_NAME}"
CHARS_TO_REMOVE_REGEX = '[!"&\(\),-./:;=?+.\n\[\]]'
PREFIX = "translate Dyula to French: "
MODEL_KWARGS = {
    "do_sample": False,
    "max_new_tokens": 32,
    "temperature": 1.0,
}


def clean_text(text: str) -> str:
    """
    Clean input text by removing special characters and converting
    to lower case.
    """
    text = re.sub(CHARS_TO_REMOVE_REGEX, " ", text.lower())
    return text.strip()


class MyModel(Model):
    """Kserve inference implementation of model."""

    def __init__(self, name: str):
        super().__init__(name)
        self.name = name
        self.model = None
        self.tokenizer = None
        self.ready = False
        self.load()

    def load(self):
        self.tokenizer = AutoTokenizer.from_pretrained('google-t5/t5-small')
        self.model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_REPO)
        self.ready = True

    def preprocess(self, payload: InferRequest, *args, **kwargs) -> str:
        raw_data = payload.inputs[0].data[0]
        prepared_data = f"{PREFIX}{clean_text(raw_data)}"
        return prepared_data

    def predict(self, data: str, *args, **kwargs) -> InferResponse:
        print(f"input data: {data}")
        inference_input = self.tokenizer(data,return_tensors='pt').input_ids
        print(f"inference input: {inference_input}")
        output = self.model.generate(inference_input, **MODEL_KWARGS)
        translation = self.tokenizer.decode(output[0], skip_special_tokens=True)
        response_id = generate_uuid()
        infer_output = InferOutput(
            name="output-0", shape=[1], datatype="STR", data=[translation]
        )
        infer_response = InferResponse(
            model_name=self.name, infer_outputs=[infer_output], response_id=response_id
        )
        return infer_response

if __name__ == "__main__":
    model = MyModel("model")
    ModelServer().start([model])