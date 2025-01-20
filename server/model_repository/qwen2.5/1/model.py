import os

# os.environ["TRANSFORMERS_CACHE"] = "/opt/tritonserver/model_repository/qwen2.5/hf-cache"

import json

import numpy as np
import torch
import transformers
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def initialize(self, args):
        self.logger = pb_utils.Logger
        self.model_config = json.loads(args["model_config"])
        self.model_params = self.model_config.get("parameters", {})
        default_hf_model = "Qwen/Qwen2.5-0.5B-Instruct"
        HUGGINGFACE_TOKEN = ""
        default_max_gen_length = "15"
        # Check for user-specified model name in model config parameters
        hf_model = self.model_params.get("huggingface_model", {}).get(
            "string_value", default_hf_model
        )
        # if "HUGGINGFACE_TOKEN" not in os.environ:
        #     print(
        #         "envvar HUGGINGFACE_TOKEN should be specified if running a restricted model like Llama2"
        #     )
        # HUGGINGFACE_TOKEN = os.environ["HUGGINGFACE_TOKEN"]

        # Check for user-specified max length in model config parameters
        self.max_output_length = int(
            self.model_params.get("max_output_length", {}).get(
                "string_value", default_max_gen_length
            )
        )

        self.logger.log_info(f"Max output length: {self.max_output_length}")
        self.logger.log_info(f"Loading HuggingFace model: {hf_model}...")
        # Assume tokenizer available for same model
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            hf_model, #token=HUGGINGFACE_TOKEN
        )

        self.pipeline = transformers.pipeline(
            "text-generation",
            model=hf_model,
            torch_dtype=torch.float16,
            tokenizer=self.tokenizer,
            device_map="auto",
            # token=HUGGINGFACE_TOKEN,
        )

    def execute(self, requests):
        responses = []
        for request in requests:
            # Assume input named "prompt", specified in autocomplete above
            input_tensor = pb_utils.get_input_tensor_by_name(request, "text_input")
            prompt = input_tensor.as_numpy()[0].decode("utf-8")

            response = self.generate(prompt)
            responses.append(response)

        return responses

    def generate(self, prompt):
        sequences = self.pipeline(
            prompt,
            do_sample=True,
            top_k=10,
            num_return_sequences=1,
            eos_token_id=self.tokenizer.eos_token_id,
            max_length=self.max_output_length,
        )

        output_tensors = []
        texts = []
        for i, seq in enumerate(sequences):
            text = seq["generated_text"]
            self.logger.log_info(f"Sequence {i+1}: {text}")
            texts.append(text)

        tensor = pb_utils.Tensor("text_output", np.array(texts, dtype=np.object_))
        output_tensors.append(tensor)
        response = pb_utils.InferenceResponse(output_tensors=output_tensors)
        return response

    def finalize(self):
        print("Cleaning up...")