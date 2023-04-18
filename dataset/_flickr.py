from string import digits, punctuation
from typing import Callable, Union

import pandas as pd
import torch
import torch.utils.data as data
from PIL import Image
from torchvision import transforms as T

from .VocabularyBag import VocabularyBag


class Flickr(data.Dataset):
    """Dataset subclass to represent the Flickr8K dataset structure.

    Loads the preprocesses the corresponding CSV to generate sparse matrices
    from the given captions for each image.

    Args:
        word_frequency_threshold (int): The minimum frequency of a word
            appearing to keep in the vocabulary.
        unknown_threshold (int): The maximum amount of <UNK> tokens a sentence can contain.
        tform (Callable): The image tranformation pipeline.
        flickr_dir (str, optional): The path to the directory containing the
            image folder and caption csv file.
        caption_file (str, optional): The file name with extension of the caption csv file.

    Attributes:
        root (str): The dataset root folder.
        data (pandas.DataFrame): The loaded captions csv file after
            preprocessing.
        vocab (VocabularyBag): The bag of words creates from the captions.
    
    """
    def __init__(
        self,
        word_frequency_threshold: int,
        unknown_threshold: int,
        tform: Callable = T.PILToTensor(),
        flickr_dir: str = "./flickr8k/",
        caption_file: str = "captions.txt",
    ) -> None:
        self.root = flickr_dir
        self.data = pd.read_csv(flickr_dir + caption_file)
        self.data["caption"] = self.data["caption"].apply(
            lambda s: "".join(c for c in s if c not in digits + punctuation)
        )
        self.__tform = tform
        self.vocab = VocabularyBag(" ".join(self.data["caption"].tolist()), word_frequency_threshold)
        self.data = self.data[
            self.data["caption"]
            .apply(self.vocab.tokenize)
            .apply(lambda token: token.count(self.vocab.word_to_idx["<UNK>"]))
            <= unknown_threshold
        ].reset_index(drop=True)
        self.data.sort_values(by="caption", key=lambda col: col.str.len(), inplace=True, ignore_index=True)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: Union[int, slice]) -> (torch.Tensor, torch.Tensor):
        img = Image.open(self.root + "images/" + self.data["image"][idx]).convert("RGB")
        caption = self.data["caption"][idx]
        img = self.__tform(img)
        tokenized_caption = self.vocab.tokenize(caption)
        return img, torch.tensor(tokenized_caption, dtype=torch.long)


class FlickrCollate:
    """A custom collate_fn for the pytorch dataloader.

    Since each batch may contain sequences of variable length,
    it was needed to pad the sequences up to the maximum length
    present at the batch.
    """
    def __init__(self, padding_idx: int):
        self.__padding_idx = padding_idx

    def __call__(self, batch: tuple[torch.Tensor, torch.Tensor]) -> (torch.Tensor, torch.Tensor):
        if len(batch) > 1:
            imgs = torch.cat(tuple(item[0].unsqueeze(0) for item in batch), dim=0)
            caption = torch.nn.utils.rnn.pad_sequence(
                [item[1] for item in batch], batch_first=True, padding_value=self.__padding_idx
            )
        else:
            imgs, caption = batch[0]
            imgs = imgs.unsqueeze(0)
        return imgs, caption
