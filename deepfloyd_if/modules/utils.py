# -*- coding: utf-8 -*-
import numpy as np
from mindspore.dataset import transforms, vision


def predict_proba(X, weights, biases):
    logits = X @ weights.T + biases
    proba = np.where(logits >= 0, 1 / (1 + np.exp(-logits)), np.exp(logits) / (1 + np.exp(logits)))
    return proba.T


def load_model_weights(path):
    model_weights = np.load(path)
    return model_weights['weights'], model_weights['biases']


def clip_process_generations(generations):
    min_size = min(generations.shape[-2:])
    return transforms.Compose([
        vision.CenterCrop(min_size),
        vision.Resize(224, interpolation=vision.Inter.BICUBIC, antialias=True),
        vision.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])(generations)
