import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import requests
import torch
from themoviedb import TMDb
from tqdm import tqdm

from datasets import dataset_factory

torch.manual_seed(42)
parser = argparse.ArgumentParser()
args = parser.parse_args()
np.set_printoptions(suppress=True)


def requestPicture(image_url, save_image_path):
    if not os.path.exists(save_image_path):
        headers = {'Connection': 'close'}
        try:
            with requests.get(url=image_url, headers=headers) as request_result:
                if request_result.status_code == 200:
                    with open(save_image_path, 'wb') as fileObj:
                        fileObj.write(request_result.content)
                    return True
        except Exception:
            return False


def find_img(movie_name):
    try:
        KeyDb = "22f10ca52f109158ac7fe064ebbcf697"
        tmdb = TMDb(key=KeyDb)
        movies = tmdb.search().movies(movie_name[:-7])
        movie_id = movies[0].id  # get first result
        movie = tmdb.movie(movie_id).details(append_to_response="images")
        img_url = "https://image.tmdb.org/t/p/original/" + movie.poster_path
        return img_url
    except:
        return ""


def main(args):
    dataset = dataset_factory(args)
    dataset = dataset.load_dataset()

    processed_path = f"./data/preprocessed/{args.dataset_code}_min_rating{args.min_rating}-min_uc{args.min_uc}-min_sc{args.min_sc}/"
    os.makedirs(processed_path + "img", exist_ok=True)

    if args.dataset_code in ["beauty", "toys", "games"]:
        with ThreadPoolExecutor(max_workers=128) as executor:
            process_list = []
            for key, value in dataset['meta_img_url'].items():
                value = value.replace("._SY300_", "")
                image_file = os.path.join(processed_path, "img", str(key)) + ".jpg"
                process = executor.submit(requestPicture, value, image_file)
                process_list.append(process)

            for _ in tqdm(as_completed(process_list), total=len(process_list)):
                pass

    if args.dataset_code in ["ml-100k"]:
        for key, value in tqdm(dataset['meta'].items()):
            image_file = os.path.join(processed_path, "img", str(key)) + ".jpg"

            if not os.path.exists(image_file):
                value = find_img(value)
                if value == "":
                    continue

                requestPicture(value, image_file)
    print("done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_code', type=str, default='llm')
    parser.add_argument('--min_rating', type=int, default=0)
    parser.add_argument('--min_uc', type=int, default=5)
    parser.add_argument('--min_sc', type=int, default=5)
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'])
    parser.add_argument('--dataset_code', type=str, default="beauty")

    args = parser.parse_args()

    main(args)
