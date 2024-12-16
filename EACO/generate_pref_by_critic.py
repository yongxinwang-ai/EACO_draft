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
random.seed(42)

def get_score(response):
    pattern = r'(\d+)'
    match = re.search(pattern, response, re.IGNORECASE)
    if match:
        return int(match.group(1))
    else:
        return 0

def image_parser(args):
    out = args.image_file.split(args.sep)
    return out


def load_image(image_file, image_corruption=False):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")

    if image_corruption:
        if random.random() > 0.5:
            image = T.Resize(size=20)(image)
        else:
            jitter = T.ColorJitter(brightness=.5, hue=.3)
            image = jitter(image)
    return image


def load_images(image_files, image_corruption=False):
    out = []
    for image_file in image_files:
        image = load_image(image_file, image_corruption)
        out.append(image)
    return out


def eval_model(args, image_corruption=False):

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
    images = load_images([image_files], image_corruption)
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
    parser.add_argument("--image-dir", type=str, default="/data/user/MSCOCO/train2014")
    parser.add_argument("--save-dir", type=str, default="pref_data_mscoco.jsonl")
    parser.add_argument("--image-file", type=str, default="/data/user/MSCOCO/val2014/COCO_val2014_000000033958.jpg")
    parser.add_argument("--query", type=str, default="Describe the image.")
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--sep", type=str, default=",")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    args = parser.parse_args()

    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path, args.model_base, model_name
    )

    prompt_list = ["Illustrate the details of the picture.",
                   "Summarize the visual content presented.",
                   "Explain what is depicted in the photograph.",
                   "Outline the key elements captured in the image.",
                   "Detail the composition and subjects within the frame.",
                   "Convey the atmosphere and mood represented in the snapshot.",
                   "Interpret the scene shown in the image.",
                   "Identify and describe the main focal points in the visual."]
    
    full_prompt = """Please provide a detailed description of the image, focusing on the following. 
    Identify the main subjects (people, animals, objects) in the image and describe what they are doing.
    Describe the setting of the image. Is it indoors or outdoors? What kind of environment or location does it depict? 
    What mood does the image convey? Are there any specific elements (such as lighting, weather, expressions) that contribute to this atmosphere? 
    Describe the dominant colors and the overall composition. How do these elements affect the image's impact?
    Point out any details or symbols that might be relevant to understanding the image's meaning or context.
    If applicable, provide interpretations of what the image might represent or communicate."""


    rate_prompt = """Review the user’s question and the corresponding response using these criteria. Points are accumulated based on the satisfaction of each
        criterion:\n\n 
        - Relevance: Is the response relevant and provides some information related to the user’s inquiry and visual information, even if it is incomplete or contains some irrelevant content?\n
        - Substantial Coverage: if the response addresses a substantial portion of the user’s question and visual information but does not completely resolve the query or provide a direct answer. \n 
        - Basic Elements: if the response answers the basic elements of the user’s question and visual information in a useful way, regardless of whether it seems to have been written by an AI Assistant or if it has elements typically found in blogs or search results. \n 
        - Clarity and Organization: if the response is clearly written from an AI Assistant’s perspective, addressing the user’s question directly and summarize the visual information comprehensively, and is well-organized and helpful, even if there is slight room for improvement in clarity, conciseness or focus. \n 
        - High Quality: for a response that is impeccably tailored to the user’s question by an AI Assistant, without extraneous information, reflecting expert knowledge, and demonstrating a high-quality, engaging, and insightful answer.\n\n After examining the user’s instruction and the response:"""


    Judge_prompt = """Review the user’s question and the corresponding response using the additive 5-point scoring system described below. Points are added based on the satisfaction of each criterion:\n\n
        - Add 1 point if the response is relevant and provides some information related to the user’s inquiry and visual information, even if it is incomplete or contains some irrelevant content.\n
        - Award another point if the response addresses a substantial portion of the user’s question and visual information but does not completely resolve the query or provide a direct answer. \n
        - Give a third point if the response answers the basic elements of the user’s question and visual information in a useful way, regardless of whether it seems to have been written by an AI Assistant or if it has elements typically found in blogs or search results. \n
        - Award a fourth point if the response is clearly written from an AI Assistant’s perspective, addressing the user’s question directly and summarize the visual information comprehensively, and is well-organized and helpful, even if there is slight room for improvement in clarity, conciseness or focus. \n
        - Add a fifth point for a response that is impeccably tailored to the user’s question by an AI Assistant, without extraneous information, reflecting expert knowledge, and demonstrating a high-quality, engaging, and insightful answer.\n\n After examining the user’s instruction and the response:
    """
    Cons_prompt = """Review the user’s question and the corresponding response using the subtractive 5-point
        scoring system described below. The initial point is 5. Points are deducted based on the satisfaction of each
        criterion:\n\n
        - Deduct 1 point if the response is irrelevant or provides no information related to the user’s inquiry and visual information.\n
        - Deduct another point if the response is completely unrelated to the user’s question and visual information, or if it is nonsensical or incoherent. \n
        - Subtract a third point if the response is factually incorrect, misleading, or contains false information, or if it is nonsensical or incoherent. \n
        - Remove a fourth point if the response is factually incorrect, misleading, or contains false information, or if it is nonsensical or incoherent. \n
        - Deduct a fifth point if the response is factually incorrect, misleading, or contains false information, or if it is nonsensical or incoherent. \n\n After examining the user’s instruction and the response:"""

    Rating_prompt = """\n Provide a concise assessment with a score from 1 to 5 for each criterion, and the scores of these criteria should be additive for a total score. Conclude with the score using the format: “score: <total points>”"
    """

    Score_prompt =        """ \n- Briefly justify your total score, up to 100 words.
        \n- Conclude with the score using the format: “score: <total points>”"""
    
    hallu_prompt_list = ["Describe the image with imaginative objects that may exist in the scene.",
                         "Enrich the description by adding hypothetical objects or characters that could be part of the scene.",
                         "Suggest and detail practical items or people that could logically inhabit the image's setting.",
                         "Incorporate elements that, though absent, would seamlessly fit into the context of the picture.",
                         "Imagine and describe additional everyday objects or activities taking place just out of frame.",
                         "Augment the scene with details of potential events or items that are plausible.",
                         "Conceive of and detail natural elements, such as weather or animals, that could realistically enter the scene. Make the description affirmative.",
                         "Invent and incorporate details of practical tools, vehicles, or gadgets that could be expected in a similar scenario."]


    directory = args.image_dir


    # load all images from coco 
    # coco = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    
    # load the 6k subset used in stic
    with open("data/6k_coco_names.json", "r") as f:
        coco = json.load(f)
    # random.shuffle(coco)

    # for i in tqdm(range(len(coco))):
    for i in tqdm(len(coco)):
        image_name = coco[i]
        args.image_file = f"{args.image_dir}/{image_name}"


        score_list = []
        output_list = []
        prompt = random.choice(prompt_list)
        for i in range(5):
            args.query = prompt
            output = eval_model(args)
            output_list.append(output)

            args.query = Judge_prompt + f"{output}" + Rating_prompt
            # args.query = Cons_prompt + f"{output}" + Score_prompt
            judgement = eval_model(args)
            score_list.append(get_score(judgement))

        max_index = score_list.index(max(score_list))
        min_index = score_list.index(min(score_list))

        if max_index == min_index:
            max_index = (max_index + 1) % 5
        
        preferred_output = output_list[max_index]
        corrupted_output = output_list[min_index]

        d = {"image": image_name,
        "prompt": prompt,
        "score_list": score_list,
        "chosen": [{"role":"user","content":prompt},{"role":"assistant","content":preferred_output}],
        "rejected": [{"role":"user","content":prompt},{"role":"assistant","content":corrupted_output}]}

        d = {"image": image_name,
        "prompt": prompt,
        "output_list": output_list}


        with open(args.save_dir,"a") as f:
            f.write(json.dumps(d))
            f.write("\n")

        


            