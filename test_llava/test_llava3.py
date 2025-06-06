import os
import torch
#from LLaVA.llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria, process_images
from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt
from transformers import TextStreamer

from habitat.core.logging import logger

model_path = "liuhaotian/llava-v1.5-13b"
load_4bit = True
load_8bit = not load_4bit
disable_torch_init()

model_name = get_model_name_from_path(model_path)
tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, None, model_name, load_8bit, load_4bit)


def load_image(image_file):
    image = Image.open(image_file).convert('RGB')
    return image


def generate_response(image, input_text):
    if 'llama-2' in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    conv = conv_templates[conv_mode].copy()
    roles = conv.roles if "mpt" not in model_name.lower() else ('user', 'assistant')

    image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].half().cuda()

    inp = input_text
    if image is not None:
        if model.config.mm_use_im_start_end:
            inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
        else:
            inp = DEFAULT_IMAGE_TOKEN + '\n' + inp

        conv.append_message(conv.roles[0], inp)
        image = None
    else:
        conv.append_message(conv.roles[0], inp)

    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            do_sample=True,
            temperature=0.2,
            max_new_tokens=2048,
            streamer=streamer,
            use_cache=True,
        )

    outputs = tokenizer.decode(output_ids[0]).strip()
    conv.messages[-1][-1] = outputs
    outputs = outputs.replace("\n\n", " ")
    return outputs

def create_description_sometimes(image_list, results_image):
    input_text1 = "# Instructions\n"\
                "You are an excellent property writer.\n"\
                "Please understand the details of the environment of this building from the pictures you have been given and explain what it is like to be in this environment as a person in this environment."

    image_descriptions = []
    for image in image_list:
        response = generate_response(image, input_text1)
        response = response[4:-4]
        image_descriptions.append(response)

    input_text2 = "# Instructions\n"\
                "You are an excellent property writer.\n"\
                "# Each_Description is a description of the building in the pictures you have entered. Please summarize these and write a description of the entire environment as if you were a person in this environment.\n"\
                "\n"\
                "# Each_Description\n"
    input_text3 = "# Notes\n"\
                "・Please summarize # Each_Description and write a description of the entire environment as if you were a person in this environment.\n"\
                "・Please write approximately 100 words.\n"\
                "・Please note that the sentences in # Each_Description are not necessarily close in distance."

    for description in image_descriptions:
        each_description = "・" + description + "\n"
        input_text2 += each_description

    input_text = input_text2 + "\n" + input_text3

    response = generate_response(results_image, input_text)
    response = response[4:-4]

    return response, image_descriptions

if __name__ == '__main__':
    
    #human_test()
    
    #model_path = "/gs/fs/tga-aklab/matsumoto/Main/model/llava-v1.5-7b/"
    model_path = "liuhaotian/llava-v1.5-13b"
    
    image_list = []
    image_folder = "/gs/fs/tga-aklab/matsumoto/Main/selected_pictures"
    for i in range(10):
        image_result_file = f"{image_folder}/{i}.png"
        logger.info(image_result_file)
        image = load_image(image_result_file)
        image_list.append(image)
    results_image = load_image("/gs/fs/tga-aklab/matsumoto/Main/selected_pictures/result_image.png")

    response, _ = create_description_sometimes(image_list, results_image)

    #plt.imshow(image)
    #plt.axis('off') 
    #plt.show()

    logger.info("#######################")
    logger.info(f"{response}")
    logger.info("FINISH !!")
