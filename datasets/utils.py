import numpy as np
import pandas as pd
from tqdm import tqdm
import urllib.request


from pathlib import Path
import zipfile
import tarfile
import sys
import hashlib

import rich.progress


def download(url, savepath):
    urllib.request.urlretrieve(url, str(savepath))
    print()


def unzip(zippath, savepath):
    print("Extracting data...")
    zip = zipfile.ZipFile(zippath)
    zip.extractall(savepath)
    zip.close()


def unziptargz(zippath, savepath):
    print("Extracting data...")
    f = tarfile.open(zippath)
    f.extractall(savepath)
    f.close()


def encrypt(fpath: str) -> str:
    with rich.progress.open(fpath, 'rb') as f:
        return hashlib.new("sha256", f.read()).hexdigest()

