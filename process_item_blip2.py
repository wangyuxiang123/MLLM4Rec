import argparse
import os
import pickle

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import Blip2Processor, Blip2ForConditionalGeneration

from datasets import dataset_factory, utils

torch.manual_seed(42)
np.set_printoptions(suppress=True)


def main(args):
    dataset = dataset_factory(args)
    dataset = dataset.load_dataset()
    meta = dataset['meta']

    print("load blip2")
    model = Blip2ForConditionalGeneration.from_pretrained(args.model_path, torch_dtype=torch.float16)
    model.to(args.device)

    dataset["meta_img_des"] = {}

    for key, value in tqdm(meta.items()):
        img_path = f"./data/preprocessed/{args.dataset_code}_min_rating{args.min_rating}-min_uc{args.min_uc}-min_sc{args.min_sc}/img/{key}.jpg"
        if not os.path.exists(img_path):
            dataset["meta_img_des"].update({key: ""})
            continue
        if args.dataset_code == "games":
            sha256 = utils.encrypt(img_path)
            if sha256 == "62dc61deb353e95e24a0ea690ea3fe2d1c527a8711b7f638633c3a978100b4e1":
                dataset["meta_img_des"].update({key: ""})
                continue

        processor = Blip2Processor.from_pretrained(args.model_path)

        image = Image.open(img_path)
        inputs = processor(images=image, return_tensors="pt").to(args.device, torch.float16)

        generated_ids = model.generate(**inputs)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

        dataset["meta_img_des"].update({key: generated_text})

    dataset_path = f"./data/preprocessed/{args.dataset_code}_min_rating{args.min_rating}-min_uc{args.min_uc}-min_sc{args.min_sc}/dataset.pkl"
    with open(dataset_path, "wb") as f:
        pickle.dump(dataset, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_code', type=str, default='llm')
    parser.add_argument('--min_rating', type=int, default=0)
    parser.add_argument('--min_uc', type=int, default=5)
    parser.add_argument('--min_sc', type=int, default=5)
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'])
    parser.add_argument('--dataset_code', type=str, default="beauty")
    parser.add_argument('--model_path', type=str, default=None)

    args = parser.parse_args()
    main(args)
