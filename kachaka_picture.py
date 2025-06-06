import os
from PIL import Image, ImageDraw
import re
import torch
import numpy as np

import clip
from sentence_transformers import util
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry


device = (
    torch.device("cuda")
    if torch.cuda.is_available()
    else torch.device("cpu")
)
# Load the clip model
clip_model, preprocess = clip.load('ViT-B/32', device)

def get_picture_list():
    # ファイルが保存されているディレクトリ
    image_directory = "./test_kachaka_select" 

    # picture_listを格納するリスト
    picture_list = []

    # ファイル名の正規表現パターン
    pattern = re.compile(rf"^(\d+)-([\d.]+)\.png$")

    # ディレクトリ内のファイルを確認
    for filename in os.listdir(image_directory):
        match = pattern.match(filename)
        if match:
            number, value = list(map(int, match.groups()[:1]))[0], float(match.group(2))
            filepath = os.path.join(image_directory, filename)
                
            # 画像を読み込む
            with Image.open(filepath) as img:
                rgb_image = np.array(img.convert("RGB"))
                #print(f"filepath:{filepath}, value:{value}")
                    
                # [RGB画像, value] をリストに追加
                picture_list.append([rgb_image, value, 0.0])

    # 結果の確認
    print(f"picture_list: {len(picture_list)}")
    return picture_list

def _count_category(taken_picture_list):
    sam = sam_model_registry["vit_b"](checkpoint="sam_model/sam_vit_b.pth")
    sam.to(device=device)

    mask_generator = SamAutomaticMaskGenerator(
        sam,
        min_mask_region_area=1000,  # 小さな領域を除外
        stability_score_thresh=0.9, # 不安定なマスクを除外
        pred_iou_thresh=0.9,        # 高精度なマスクのみ保持
    ) 

    for i in range(len(taken_picture_list)):
        image = taken_picture_list[i][0]
        pic_val = taken_picture_list[i][1]
                
        masks = mask_generator.generate(np.array(image))
                
        sorted_anns = sorted(masks, key=(lambda x: x['area']), reverse=True)
                
        image_size = image.shape[0] * image.shape[1]
        category_count = 0
        for ann in sorted_anns:
            area = ann['area']
            if area < image_size * 0.01:
                break
                    
            category_count += 1
                    
        taken_picture_list[i][2] = pic_val * category_count
        
    return taken_picture_list

def _create_new_image_embedding(obs):
    image = Image.fromarray(obs)
    image = preprocess(image)
    image = torch.tensor(image).to(device).unsqueeze(0)
    embetting = clip_model.encode_image(image).float()
    return embetting

def _decide_save(emd, results):
    for i in range(len(results)):
        check_emb = _create_new_image_embedding(results[i][0])

        sim = util.pytorch_cos_sim(emd, check_emb).item()
        if sim >= 0.9:
            return False
    return True

def _select_pictures(taken_picture_list):
    # Segment Anythingで物体数をpi_valにかける
    taken_picture_list = _count_category(taken_picture_list)
        
    sorted_picture_list = sorted(taken_picture_list, key=lambda x: x[2], reverse=True)
    results = []
    i = 0
    while True:
        if len(results) == 10:
            break
        if i == len(sorted_picture_list):
            break
        emd = _create_new_image_embedding(sorted_picture_list[i][0])
        is_save = _decide_save(emd, results)

        if is_save == True:
            results.append(sorted_picture_list[i])
        i += 1

    return results

def _create_results_image(picture_list):
    images = []
    
    if len(picture_list) == 0:
        return None

    for i in range(10):
        idx = i%len(picture_list)
        images.append(Image.fromarray(picture_list[idx][0]))

    width, height = images[0].size
    result_width = width * 5
    result_height = height * 2
    result_image = Image.new("RGB", (result_width, result_height))

    for i, image in enumerate(images):
        x_offset = (i % 5) * width
        y_offset = (i // 5) * height
        result_image.paste(image, (x_offset, y_offset))
        
    draw = ImageDraw.Draw(result_image)
    for x in range(width, result_width, width):
        draw.line([(x, 0), (x, result_height)], fill="black", width=7)
    for y in range(height, result_height, height):
        draw.line([(0, y), (result_width, y)], fill="black", width=7)

    return result_image


if __name__ == "__main__":
    picture_list = get_picture_list()
    selected_pictures = _select_pictures(picture_list)
    print(f"selected_pictures: {len(selected_pictures)}")

    for i in range(len(selected_pictures)):
        value1 = selected_pictures[i][1]
        value2 = selected_pictures[i][2]
        print(f"{value1}, {value2}")

        picture = Image.fromarray(np.uint8(selected_pictures[i][0]))
        picture.save(f"./selected_pictures/{i}.png")

    result_image = _create_results_image(selected_pictures)
    result_image.save(f"selected_pictures/result_image.png")


