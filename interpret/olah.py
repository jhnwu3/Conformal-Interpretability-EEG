import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from models.st_transformer import *
from abc import ABC, abstractmethod
import cv2

# for now just do the st transformer
def visualize_classification_weights(transformer :STTransformer, path="fig/sttransformer_classification_weights.png"):
    # get the last transformer layer
    ff = transformer.classification
    linear_layers = [layer for layer in ff if isinstance(layer, nn.Linear)]
    # get the last MLP layer
    mlp = linear_layers[-1]
    plt.figure(figsize=(10,2))
    plt.imshow(mlp.cpu().weight.detach().numpy(), cmap='plasma')
    plt.colorbar()
    # Remove white space around the plot
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # Remove the axes
    # plt.axis('off')
    plt.savefig(path)

def visualize_mlp_weights(model :STTransformer, path="fig/sttransformer_mlp_weights.png"):
    ff = model.transformer.transformer_blocks[-1].ff # last feedforward
    mlp = [layer for layer in ff if isinstance(layer, nn.Linear)][-1] # last mlp layer
    plt.figure(figsize=(10,2))
    plt.imshow(mlp.cpu().weight.detach().numpy(), cmap='plasma')
    plt.colorbar()
    # Remove white space around the plot
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(path)

# sample is a B x C x T tensor
# returns a feature from a sparse autoencoder
def generate_feature(sample, model):
    # we need to rewrite the weights to allow the model to retrieve the logits.
    return None


def generate_all_features(dataloader, model):
    return None