import argparse
import os

import numpy as np
from tqdm import tqdm
import json
import torch
import torchvision
from PIL import Image
import litellm

# Grounding DINO
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

# segment anything
from segment_anything import (
    build_sam,
    build_sam_hq,
    SamPredictor
) 
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Recognize Anything Model & Tag2Text
from ram.models import ram
from ram import inference_ram
import torchvision.transforms as TS

# ChatGPT or nltk is required when using tags_chineses
# import openai
# import nltk

def load_models(config_file, grounded_checkpoint, ram_checkpoint, sam_hq_checkpoint, sam_checkpoint, use_sam_hq, device):
    # load model
    grounded_model = load_grounded_model(config_file, grounded_checkpoint, device=device)
    transform, ram_model = load_ram_model(ram_checkpoint, device)
    sam_model = load_sam_model(sam_hq_checkpoint, sam_checkpoint, use_sam_hq, device)
    return grounded_model, ram_model, sam_model, transform


def load_sam_model(sam_hq_checkpoint, sam_checkpoint, use_sam_hq, device):
    # initialize SAM
    if use_sam_hq:
        print("Initialize SAM-HQ Predictor")
        predictor = SamPredictor(build_sam_hq(checkpoint=sam_hq_checkpoint).to(device))
    else:
        predictor = SamPredictor(build_sam(checkpoint=sam_checkpoint).to(device))
    return predictor

def load_ram_model(ram_checkpoint, device):
    # initialize Recognize Anything Model
    normalize = TS.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform = TS.Compose([
                    TS.Resize((384, 384)),
                    TS.ToTensor(), normalize
                ])
    
    # load model
    ram_model = ram(pretrained=ram_checkpoint,
                                        image_size=384,
                                        vit='swin_l')
    # threshold for tagging
    # we reduce the threshold to obtain more tags
    ram_model.eval()

    ram_model = ram_model.to(device)
    return transform, ram_model

def load_image(image_path):
    # load image
    image_pil = Image.open(image_path).convert("RGB")  # load image

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image_pil, image


def check_tags_chinese(tags_chinese, pred_phrases, max_tokens=100, model="gpt-3.5-turbo"):
    object_list = [obj.split('(')[0] for obj in pred_phrases]
    object_num = []
    for obj in set(object_list):
        object_num.append(f'{object_list.count(obj)} {obj}')
    object_num = ', '.join(object_num)
    print(f"Correct object number: {object_num}")

    if openai_key:
        prompt = [
            {
                'role': 'system',
                'content': 'Revise the number in the tags_chinese if it is wrong. ' + \
                           f'tags_chinese: {tags_chinese}. ' + \
                           f'True object number: {object_num}. ' + \
                           'Only give the revised tags_chinese: '
            }
        ]
        response = litellm.completion(model=model, messages=prompt, temperature=0.6, max_tokens=max_tokens)
        reply = response['choices'][0]['message']['content']
        # sometimes return with "tags_chinese: xxx, xxx, xxx"
        tags_chinese = reply.split(':')[-1].strip()
    return tags_chinese


def load_grounded_model(model_config_path, model_checkpoint_path, device):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(load_res)
    _ = model.eval()
    return model


def get_grounding_output(model, image, caption, box_threshold, text_threshold,device="cpu"):
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
    logits.shape[0]

    # filter output
    logits_filt = logits.clone()
    boxes_filt = boxes.clone()
    filt_mask = logits_filt.max(dim=1)[0] > box_threshold
    logits_filt = logits_filt[filt_mask]  # num_filt, 256
    boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
    logits_filt.shape[0]

    # get phrase
    tokenlizer = model.tokenizer
    tokenized = tokenlizer(caption)
    # build pred
    pred_phrases = []
    scores = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
        pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
        scores.append(logit.max().item())

    return boxes_filt, torch.Tensor(scores), pred_phrases


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax, label):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2)) 
    ax.text(x0, y0, label)


def save_mask_data(output_dir, tags_chinese, mask_list, box_list, label_list):
    value = 0  # 0 for background

    mask_img = torch.zeros(mask_list.shape[-2:])
    for idx, mask in enumerate(mask_list):
        mask_img[mask.cpu().numpy()[0] == True] = value + idx + 1
    plt.figure(figsize=(10, 10))
    plt.imshow(mask_img.numpy())
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, 'mask.jpg'), bbox_inches="tight", dpi=300, pad_inches=0.0)

    json_data = {
        'tags_chinese': tags_chinese,
        'mask':[{
            'value': value,
            'label': 'background'
        }]
    }
    for label, box in zip(label_list, box_list):
        value += 1
        name, logit = label.split('(')
        logit = logit[:-1] # the last is ')'
        json_data['mask'].append({
            'value': value,
            'label': name,
            'logit': float(logit),
            'box': box.numpy().tolist(),
        })
    with open(os.path.join(output_dir, 'label.json'), 'w') as f:
        json.dump(json_data, f)


def save_mask_data_v2(output_dir, img_id, mask_list, box_list, label_list):
    value = 0  # 0 for background

    mask_img = torch.zeros(mask_list.shape[-2:])
    for idx, mask in enumerate(mask_list):
        mask_img[mask.cpu().numpy()[0] == True] = value + idx + 1

    # Convert the mask to NumPy array
    mask_array = mask_img.numpy().astype(np.uint8)  # Convert to uint8 for saving
    image = Image.fromarray(mask_array, mode="L")  # 'L' mode for grayscale
    save_path = os.path.join(output_dir, f'{img_id}.png')
    image.save(save_path)

    json_data = {
        'mask_path': save_path, 
        'mask': []
    }
    for label, box in zip(label_list, box_list):
        value += 1
        name, logit = label.split('(')
        logit = logit[:-1] # the last is ')'
        json_data['mask'].append({
            'value': value,
            'label': name,
        })
    return img_id, json_data

def infer_img(image_path, output_dir, ram_model, grounded_model, sam_model, box_threshold, text_threshold, iou_threshold, device):
    img_id = int(image_path.split('/')[-1].split('.')[0])
    # load image
    image_pil, image = load_image(image_path)
    # visualize raw image
    image_pil.save(os.path.join(output_dir, "raw_image.jpg"))

    raw_image = image_pil.resize((384, 384))
    raw_image  = transform(raw_image).unsqueeze(0).to(device)

    res = inference_ram(raw_image, ram_model)

    # Currently ", " is better for detecting single tags
    # while ". " is a little worse in some case
    tags = res[0].replace(' |', ',')
    print("Image Tags: ", res[0])

    # run grounding dino model
    boxes_filt, scores, pred_phrases = get_grounding_output(grounded_model, image, tags, box_threshold, text_threshold, device=device)

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    sam_model.set_image(image)

    size = image_pil.size
    H, W = size[1], size[0]
    for i in range(boxes_filt.size(0)):
        boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
        boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
        boxes_filt[i][2:] += boxes_filt[i][:2]

    boxes_filt = boxes_filt.cpu()
    # use NMS to handle overlapped boxes
    print(f"Before NMS: {boxes_filt.shape[0]} boxes")
    nms_idx = torchvision.ops.nms(boxes_filt, scores, iou_threshold).numpy().tolist()
    boxes_filt = boxes_filt[nms_idx]
    pred_phrases = [pred_phrases[idx] for idx in nms_idx]
    print(f"After NMS: {boxes_filt.shape[0]} boxes")
    # tags_chinese = check_tags_chinese(tags_chinese, pred_phrases)
    # print(f"Revise tags_chinese with number: {tags_chinese}")

    transformed_boxes = sam_model.transform.apply_boxes_torch(boxes_filt, image.shape[:2]).to(device)

    masks, _, _ = sam_model.predict_torch(
        point_coords = None,
        point_labels = None,
        boxes = transformed_boxes.to(device),
        multimask_output = False,
    )

    # # draw output image
    # plt.figure(figsize=(10, 10))
    # plt.imshow(image)
    # for mask in masks:
    #     show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
    # for box, label in zip(boxes_filt, pred_phrases):
    #     show_box(box.numpy(), plt.gca(), label)

    # # plt.title('RAM-tags' + tags + '\n' + 'RAM-tags_chineseing: ' + tags_chinese + '\n')
    # plt.axis('off')
    # plt.savefig(
    #     os.path.join(output_dir, "automatic_label_output.jpg"), 
    #     bbox_inches="tight", dpi=300, pad_inches=0.0
    # )

    img_id, json_data = save_mask_data_v2(output_dir, img_id, masks, boxes_filt, pred_phrases)

    return {'id': img_id, 'mask': json_data}
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser("Grounded-Segment-Anything Demo", add_help=True)
    parser.add_argument("--config", type=str, required=True, help="path to config file")
    parser.add_argument(
        "--ram_checkpoint", type=str, required=True, help="path to checkpoint file"
    )
    parser.add_argument(
        "--grounded_checkpoint", type=str, required=True, help="path to checkpoint file"
    )
    parser.add_argument(
        "--sam_checkpoint", type=str, required=True, help="path to checkpoint file"
    )
    parser.add_argument(
        "--sam_hq_checkpoint", type=str, default=None, help="path to sam-hq checkpoint file"
    )
    parser.add_argument(
        "--use_sam_hq", action="store_true", help="using sam-hq for prediction"
    )
    parser.add_argument("--input_folder", type=str, required=True, help="path to image file")
    parser.add_argument("--split", default=",", type=str, help="split for text prompt")
    parser.add_argument("--openai_key", type=str, help="key for chatgpt")
    parser.add_argument("--openai_proxy", default=None, type=str, help="proxy for chatgpt")
    parser.add_argument(
        "--output_dir", "-o", type=str, default="outputs", required=True, help="output directory"
    )

    parser.add_argument("--box_threshold", type=float, default=0.25, help="box threshold")
    parser.add_argument("--text_threshold", type=float, default=0.2, help="text threshold")
    parser.add_argument("--iou_threshold", type=float, default=0.5, help="iou threshold")

    parser.add_argument("--device", type=str, default="cpu", help="running on cpu only!, default=False")
    args = parser.parse_args()

    # cfg
    config_file = args.config  # change the path of the model config file
    ram_checkpoint = args.ram_checkpoint  # change the path of the model
    grounded_checkpoint = args.grounded_checkpoint  # change the path of the model
    sam_checkpoint = args.sam_checkpoint
    sam_hq_checkpoint = args.sam_hq_checkpoint
    use_sam_hq = args.use_sam_hq
    img_folder = args.input_folder
    split = args.split
    openai_key = args.openai_key
    openai_proxy = args.openai_proxy
    output_dir = args.output_dir
    box_threshold = args.box_threshold
    text_threshold = args.text_threshold
    iou_threshold = args.iou_threshold
    device = args.device
    grounded_model, ram_model, sam_model, transform = load_models(config_file, grounded_checkpoint, ram_checkpoint, sam_hq_checkpoint, sam_checkpoint, use_sam_hq, device)
    
    # make dir
    os.makedirs(output_dir, exist_ok=True)
    mask_fig_dir = os.path.join(output_dir, 'fig')
    os.makedirs(mask_fig_dir, exist_ok=True)

    # ChatGPT or nltk is required when using tags_chineses
    # openai.api_key = openai_key
    # if openai_proxy:
        # openai.proxy = {"http": openai_proxy, "https": openai_proxy}

    img_files = os.listdir(img_folder)
    list_img = [os.path.join(img_folder, item) for item in img_files]

    list_all = []
    for img in tqdm(list_img):
        dict_res = infer_img(img, mask_fig_dir, ram_model, grounded_model, sam_model, box_threshold, text_threshold, iou_threshold, device)
        list_all.append(dict_res)

    # Save the results to a JSON file
    with open(os.path.join(output_dir, f"Segment_info.json"), "w") as f:
        json.dump(list_all, f, indent=4)
        
