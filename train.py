#!/usr/local/bin/python
# -*- coding: utf-8 -*-

import os
from pprint import pprint

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torchvision import models

from dataset import Flickr, FlickrCollate
from hypers import parse_args
from models import ImageCaptioningModel

dev = "cuda" if torch.cuda.is_available() else "cpu"

hyperparams = parse_args()
print("Hyperparameters:")
pprint(vars(hyperparams))

ds = Flickr(
    hyperparams.word_frequency_thresh,
    hyperparams.unknown_thresh,
    tform=models.ResNet152_Weights.IMAGENET1K_V1.transforms(),
)

collate = FlickrCollate(padding_idx=ds.vocab.word_to_idx["<PAD>"])

loader = data.DataLoader(
    ds,
    batch_size=hyperparams.batch_size,
    collate_fn=collate,
    pin_memory=True,
    shuffle=True,
)

model = ImageCaptioningModel(
    hyperparams.features, len(ds.vocab), hyperparams.hidden_state_size, hyperparams.lstm_layers
).to(dev)
criterion = nn.CrossEntropyLoss(ignore_index=ds.vocab.word_to_idx["<PAD>"]).to(dev)
optimizer = optim.Adam(model.parameters(), lr=hyperparams.learning_rate)

for epoch in range(hyperparams.epochs):
    model.train()
    for batch, (imgs, captions) in enumerate(loader):
        imgs, captions = imgs.to(dev), captions.to(dev)
        o_preds, o_logits = model(imgs, captions=captions[:, :-1])

        loss = criterion(o_logits.reshape(-1, o_logits.shape[2]), captions.reshape(-1))
        acc = torch.sum((o_preds == captions)) / (captions.shape[0] * captions.shape[1]) * 100
        print(f"Epoch {epoch:>4} Batch {batch:>5} : loss := {loss:>10.2f} accuracy := {acc:>10.2f}%")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

os.makedirs(os.path.dirname("./pretrained/model.pth"), exist_ok=True)
torch.save(
    model.state_dict(),
    f"./pretrained/model-f{hyperparams.features}-h{hyperparams.hidden_state_size}-l{hyperparams.lstm_layers}-lr{hyperparams.learning_rate}.pth",
)
