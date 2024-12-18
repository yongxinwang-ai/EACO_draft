import argparse
import torch
import torchvision.transforms as T
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import random
import json

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)
from llava.eval.run_llava import eval_model

import requests
from PIL import Image
from io import BytesIO
import re
import os

from tqdm import tqdm 


def image_parser(args):
    out = args.image_file.split(args.sep)
    return out


def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")

    return image


def load_images(image_files):
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out


def eval_model(args):

    qs = args.query
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    if IMAGE_PLACEHOLDER in qs:
        if model.config.mm_use_im_start_end:
            qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
        else:
            qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
    else:
        if model.config.mm_use_im_start_end:
            qs = image_token_se + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "mistral" in model_name.lower():
        conv_mode = "mistral_instruct"
    elif "v1.6-34b" in model_name.lower():
        conv_mode = "chatml_direct"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print(
            "[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(
                conv_mode, args.conv_mode, args.conv_mode
            )
        )
    else:
        args.conv_mode = conv_mode

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    image_files = args.image_file 
    images = load_images([image_files])
    image_sizes = [x.size for x in images]
    images_tensor = process_images(
        images,
        image_processor,
        model.config
    ).to(model.device, dtype=torch.float16)

    input_ids = (
        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .cuda()
    )

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=images_tensor,
            image_sizes=image_sizes,
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    return outputs


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="liuhaotian/llava-v1.6-mistral-7b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-file", type=str, default="/data/yihe/COCO/val2014/COCO_val2014_000000033958.jpg")
    parser.add_argument("--query", type=str, default="Describe the image.")
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--sep", type=str, default=",")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=1024)

    parser.add_argument("--image-dir", type=str, default="/data1/yihedeng/image_data/")
    parser.add_argument("--save-dir", type=str, default="image_description.jsonl")
    parser.add_argument("--adapter-path", type=str, default=None)
    args = parser.parse_args()

    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path, args.model_base, model_name
    )
    if args.adapter_path is not None:
        model.load_adapter(args.adapter_path)
        print("adapter loaded!")


    full_prompt = """Please provide a detailed description of the image, focusing on the following. 
    Identify the main subjects (people, animals, objects) in the image and describe what they are doing.
    Describe the setting of the image. Is it indoors or outdoors? What kind of environment or location does it depict? 
    What mood does the image convey? Are there any specific elements (such as lighting, weather, expressions) that contribute to this atmosphere? 
    Describe the dominant colors and the overall composition. How do these elements affect the image's impact?
    Point out any details or symbols that might be relevant to understanding the image's meaning or context.
    If applicable, provide interpretations of what the image might represent or communicate."""
    
    prompt_list = ["Illustrate the details of the picture.",
                   "Summarize the visual content presented.",
                   "Explain what is depicted in the photograph.",
                   "Outline the key elements captured in the image.",
                   "Detail the composition and subjects within the frame.",
                   "Convey the atmosphere and mood represented in the snapshot.",
                   "Interpret the scene shown in the image.",
                   "Identify and describe the main focal points in the visual."]


    directory = args.image_dir
    with open('data/mixed_5k.json', 'r') as f:
       coco = json.load(f)

    image_names = [x['image'] for x in coco]
    query_list = [x['conversations'][0]['value'].replace('','<image>\n') for x in coco]
    print(len(image_names), image_names[0])

    # parallelize the task on multiple gpus to speed up the process
    for i in tqdm(range(3750, len(image_names))):
        args.image_file = f"{directory}/{image_names[i]}"
        args.query = random.choice(prompt_list)
        output = eval_model(args)

        d = {"image": args.image_file, 
            "prompt":args.query,
            "description":output}
        
        with open(args.save_dir,"a") as f:
            f.write(json.dumps(d))
            f.write("\n")