"""This modules contains the implementations of the Encoder, Decoder and Neural Image Captioning Models."""
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torchvision import models


class Encoder(nn.Module):
    """Image-to-feature vector encoder network.

    Using ResNet152, a DCNN is used for transfer learning by replacing the last
    Fully Connected layer with a new one, which will output a feature vector of
    any given image.

    Attributes:
        deep_cnn (models.ResNet): The pretrained ResNet152.
        relu (nn.Module): A ReLU layer introduced after the ResNet.
        dropout (nn.Module): A Dropout layer introduced after the ReLU.

    """

    def __init__(self, out_features: int) -> None:
        """Initializes an Encoder object.

        Args:
            out_features (int): The amount of features the new Fully Connected
                layer will output.

        """
        super().__init__()
        self.deep_cnn = models.resnet152(weights=models.ResNet152_Weights.DEFAULT)
        for param in self.deep_cnn.parameters():
            param.requires_grad = False
        self.deep_cnn.fc = nn.Linear(self.deep_cnn.fc.in_features, out_features)
        self.deep_cnn.fc.weight.requires_grad = True
        self.deep_cnn.fc.bias.requires_grad = True
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """Produces a feature vector from a given image.

        Args:
            img (torch.Tensor): An image to encode.

        Returns:
            torch.Tensor: The computed feature vector.

        """
        return self.dropout(self.relu(self.deep_cnn(img)))


class Decoder(nn.Module):
    """Decodes a feature vector into a sparse matrix.

    Given a vector representation of an image, outputs a sparse matrix
    representing the generated caption in one-hot embedding.

    Attributes:
        embed (nn.Embedding): An embedding layer.
        dropout (nn.Dropout): A dropout layer.
        lstm (nn.LSTM): The LSTM decoding network.
        fc (nn.Linear): A fully connected layer that maps the lstm output to
            the vocabulary size.
        sfm (nn.Softmax): A softmax layer.

    """

    def __init__(self, vocab_size: int, in_features: int, hidden_state_size: int, num_layers: int) -> None:
        """Initializes a Decoder object.

        Args:
            vocab_size (int): The size of the bag of words.
            in_features (int): The size of the feature vector.
            hidden_state_size (int): The size of the LSTM hidden state.
            num_layers (int): Amount of LSTMCells to use.

        """
        super().__init__()
        self.embed = nn.Embedding(vocab_size, in_features)
        self.dropout = nn.Dropout()
        self.lstm = nn.LSTM(in_features, hidden_state_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_state_size, vocab_size)
        self.sfm = nn.Softmax(dim=-1)

    def forward(
        self,
        features: torch.Tensor,
        /,
        eos: Optional[int] = None,
        captions: Optional[torch.Tensor] = None,
        length: Optional[int] = 30,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generates a caption given a feature vector.

        When in training, captions must be supplemented. When in validation
        mode the <EOS> token index must be given along with the max
        sequence length if the predefined is too much or too little.
        The sparse matrix output is greedily generated. The generation
        will continue until an <EOS> token is generated up the specified max.

        Args:
            features (torch.Tensor): The feature vector.
            eos (Optional[int], optional): The index of the <EOS> token.
                Defaults to None.
            captions (Optional[torch.Tensor], optional): The ground truth
                caption for each feature vector. Defaults to None.
            length (Optional[int], optional): The max length of the generated
                caption. Defaults to 30.
            in_prod (bool, optional): If model is in production mode.
                Defaults to False.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: The predictions and the logits
                of the network output.

        """
        if self.training:
            x = torch.cat((features.unsqueeze(1), self.dropout(self.embed(captions))), dim=1)
            o_logits, _ = self.lstm(x)
            o_logits = self.fc(o_logits)
            o_preds = self.sfm(o_logits).argmax(-1)
            output = (o_preds, o_logits)
        else:
            o_logits, (h_n, c_n) = self.lstm(features)
            # logit output -> lstm input at t+1
            logits = self.fc(o_logits)
            preds = self.sfm(logits).argmax(-1)
            x = self.embed(preds)

            for _ in range(length - 1):
                o_logits, (h_n, c_n) = self.lstm(x, (h_n, c_n))
                logits = torch.cat((logits, self.fc(o_logits)), dim=0)
                preds = torch.cat((preds, self.sfm(logits[-1]).argmax(-1).unsqueeze(0)), dim=0)
                x = self.embed(preds[-1].unsqueeze(0))
                if preds[-1] == eos:
                    break
            output = (preds, logits)
        return output


class ImageCaptioningModel(nn.Module):
    """Neural Image Captioning model.

    An Encoder-Decoder based model which takes images as inputs and generates
    captions for them.

    Attributes:
        encoder (Encoder): The encoder network.
        decoder (Decoder): The decoder network.

    """

    def __init__(self, features: int, vocab_size: int, hidden_state_size: int, lstm_layers: int) -> None:
        """Initializes an ImageCaptioningModel object.

        Args:
            features (int): The amount of features the encoder network should
                produce for each image.
            vocab_size (int): The size of the bag of words.
            hidden_state_size (int): The size of the LSTM hidden state in the
                decoder network.
            lstm_layers (int): The amount of LSTMCells used in the LSTM of the
                decoder network.

        """
        super().__init__()
        self.encoder = Encoder(features)
        self.decoder = Decoder(vocab_size, features, hidden_state_size, lstm_layers)

    def forward(
        self, imgs: torch.Tensor, captions: Optional[torch.Tensor] = None, eos: Optional[int] = None, length: int = 50
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generates a caption given an image.

        Args:
            imgs (torch.Tensor): The image(s) to generate caption for.
            captions (Optional[torch.Tensor], optional): The reference caption
                for each image. Must be supplied when training. Defaults to None.
            eos (Optional[int], optional): The <EOS> token index. Must be
                supplied in validation mode. Defaults to None.
            length (int, optional): The maximum length of the sequence to
                generated. Defaults to 50.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The generated sequence(s) in a
                sparse matrix format and the corresponding logits.
        """
        features = self.encoder(imgs)
        if self.training:
            outputs = self.decoder(features, captions=captions)
        else:
            outputs = self.decoder(features, eos=eos, length=length)
        return outputs
