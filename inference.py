#!/usr/local/bin/python
# -*- coding: utf-8 -*-
import re
from pathlib import Path

import torch
from matplotlib import pyplot as plt
from PIL import Image
from torchvision import models as M

from dataset import Flickr
from hypers import parse_args
from models import ImageCaptioningModel

hyperparams = parse_args()

PATH = "./pretrained/model-f512-h512-l1-lr0.0001.pth"

o = re.findall(r"(\d+\.?\d*)", PATH)
o.pop(-1)  # remove learning rate, unneeded for inference
o = [int(num) for num in o]

dev = "cuda" if torch.cuda.is_available() else "cpu"

ds = Flickr(hyperparams.word_frequency_thresh, hyperparams.unknown_thresh)

model = ImageCaptioningModel(o[0], len(ds.vocab), o[1], o[2])
model.load_state_dict(torch.load(PATH))
model = model.to(dev)
model.eval()

tform = M.ResNet152_Weights.IMAGENET1K_V1.transforms()

fig = plt.gcf()

with torch.no_grad():
    while True:
        img_path = input("Write the directory path of the image or exit: ")
        if img_path.lower() == "exit":
            print("Exiting.")
            break

        f = Path(img_path)
        if not f.exists() or f.is_dir():
            print("Invalid file path.")
            continue

        img = Image.open(img_path)
        x = tform(img).unsqueeze(0).to(dev)

        caption_pred, _ = model(x, eos=ds.vocab.word_to_idx["<EOS>"], length=20)
        caption = ds.vocab.stringify(caption_pred.to("cpu").tolist())

        plt.imshow(img)
        plt.title(caption)
        plt.axis("off")
        plt.show(block=False)
        plt.figure()
