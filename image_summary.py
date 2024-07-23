from transformers import (
    AutoProcessor,
    LlavaForConditionalGeneration,
    BitsAndBytesConfig,
)

import ray

import psutil

import torch

import cv2

from PIL import Image

VISION_MODEL_NAME = "llava-hf/llava-1.5-7b-hf"


@ray.remote(num_cpus=psutil.cpu_count(), num_gpus=1)
class ImageSummariser(object):

    def __init__(self):

        return

    def actual_init(self):

        self.llava_image_summary_prompt = "USER: <image>\nSummarize this image using the following context. {context}\nASSISTANT: "

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        self.vision_model = LlavaForConditionalGeneration.from_pretrained(
            VISION_MODEL_NAME,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
            quantization_config=bnb_config,
            device_map="auto",
        )

        self.vision_processor = AutoProcessor.from_pretrained(VISION_MODEL_NAME)

        return

    def summarise_image(self, image_path: str, context: str) -> dict:

        question = self.llava_image_summary_prompt.format(context=context)

        image = Image.open(image_path)

        inputs = self.vision_processor(question, image, return_tensors="pt").to(
            0, torch.float16
        )

        output = self.vision_model.generate(
            **inputs, max_new_tokens=100, do_sample=False
        )

        response = self.vision_processor.decode(output[0][2:], skip_special_tokens=True)

        answer = response.partition("ASSISTANT:")[-1].replace("\n", "")

        return answer
