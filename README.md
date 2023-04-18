# Neural Image Captioning Model
An implementation of [Show and Tell: Lessons learned from the 2015 MSCOCO Image Captioning Challenge](https://arxiv.org/abs/1609.06647) utilizing the pretrained ResNet152 Deep CNN model for the encoding part.

Without hyperparameter tuning and with greedy sequence generation even in training, the model performs well when supplied images from the training dataset, while it is adequate when giving images it has not seen before. Better preprocessing and hyperparameter tuning could make the model a lot better than the supplied pretrained one. Using sampling instead of argmax in the training sequence generation should also yield better results, along with Beam Search for inference.

The dataset used was downloaded from [here](https://www.kaggle.com/datasets/adityajn105/flickr8k). Using another dataset, such as the MS COCO as specified in the paper, should also give better generalization to unseen data.
