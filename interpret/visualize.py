# taken from Jathurshan
import math
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
from torch.nn import Module
from torch.nn import MultiheadAttention
from torch.nn import ModuleList
from torch.nn.init import xavier_uniform_
from torch.nn import Dropout
from torch.nn import Linear
from torch.nn import LayerNorm, BatchNorm1d
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce

from typing import Dict, List, Optional, Tuple
import numpy as np
import subprocess
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
from IPython import display
from ipywidgets import  interactive, IntSlider
from data import *



# returns a tuple (signals,cams)
def interpret_dataset(interpreter, test_loader):
    # get back the cams, and corresponding signals
    batches = []
    cams = []
    labels = []
    for batch, label in test_loader:
        batches.append(batch)
        cam = interpreter.get_cam(batch, class_index=label)
        cams.append(cam)
        labels.append(label)
        # print(label)
    cams = torch.cat(cams)
    batches = torch.cat(batches)
    labels = torch.cat(labels)
    return batches, cams, labels

# get back the cams for the indices in the prediction set generated
# note that the inputs must still be batched
def interpret_predictions(interpreter, input, prediction_set):
    predicted_cams = []
    for i in range(prediction_set.size()[0]):
        cam = interpreter.get_cam(input, prediction_set[i])
        predicted_cams.append(cam)
    return torch.cat(predicted_cams)

def plot_signal_attentions(x,y,dydx,axs,axs_no=0, title_font_size=10, norm=None):
    '''
    Plot the attention scores on the EEG and EOG signals
    '''

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    
    # Create a continuous norm to map from data points to colors
    if norm == None:
        norm = plt.Normalize(dydx.min(), dydx.max())
    # norm = plt.Normalize(dydx.min(), dydx.max())
    lc = LineCollection(segments, cmap='Reds', norm=norm) #, norm=norm)
    # Set the values used for colormapping
    lc.set_array(dydx)
    lc.set_linewidth(2)

    line = axs[axs_no].add_collection(lc)
    axs[axs_no].set_title(f'Channel {axs_no}', fontsize=title_font_size)
    axs[axs_no].set_xlabel(f'Time',fontsize = title_font_size - 4)
    axs[axs_no].xaxis.set_tick_params(labelbottom=False)
    axs[axs_no].yaxis.set_tick_params(labelleft=False)
    
    # Hide X and Y axes tick marks
    axs[axs_no].set_xticks([])
    axs[axs_no].set_yticks([])
    axs[axs_no].set_xlim(x.min(), x.max())
    axs[axs_no].set_ylim(y.min()-0.2,y.max()+0.2)

    return line 


# NEEDS TO BE a 1 x C x T signal tensor
# Take in signal : ndarray C x T
# cam: ndarray C x T
def visualize_signal(signal, cam, nCols=4, title=""):
    nChannels = cam.shape[0]
    nRows = nChannels // nCols
    t = np.arange(0, cam.shape[1], 1)
    # four column channel plots
    fig, ax = plt.subplots(nRows, nCols, figsize=(16,16))
    ax = ax.reshape(-1)
    common_norm = plt.Normalize(cam.min(), cam.max())
    line = None
    for i in range(nChannels):
        line = plot_signal_attentions(t, signal[i], cam[i], ax, i, norm=common_norm)
        
    fig.colorbar(line, ax=ax, shrink=0.7)
    plt.suptitle(title, fontsize=16)
    
    plt.show()

def visualize_signal_no_show(signal, cam, nCols=4, title=""):
    nChannels = cam.shape[0]
    nRows = nChannels // nCols
    t = np.arange(0, cam.shape[1], 1)
    # four column channel plots
    fig, ax = plt.subplots(nRows, nCols, figsize=(16,16))
    ax = ax.reshape(-1)
    common_norm = plt.Normalize(cam.min(), cam.max())
    for i in range(nChannels):
        plot_signal_attentions(t, signal[i], cam[i], ax, i, norm=common_norm)
    plt.suptitle(title, fontsize=16)

# cams of prediction set should be a len(pred_set) x C x T
# all inputs are ndarrays
def visualize_prediction_set(signal, cams_of_prediction_set, prediction_set):
    nCams = cams_of_prediction_set.shape[0] - 1
    select_data = IntSlider(min=0, max=nCams, step=1, value=0, description="Class Index")
  
    def update_plot(class_no):
        display.clear_output(wait=True)
        visualize_signal_no_show(signal, cams_of_prediction_set[class_no], title=f'Class {prediction_set[class_no]}')
    
    int_plot = interactive(update_plot, class_no=select_data)
    return int_plot


def visualize_dataset(signals, cams):
    nCams = cams.shape[0] - 1
    select_data = IntSlider(min=0, max=nCams, step=1, value=0, description="Class Index")
  
    def update_plot(class_no):
        display.clear_output(wait=True)
        visualize_signal_no_show(signals[class_no], cams[class_no], title=f'Sample {class_no}')
    
    int_plot = interactive(update_plot, class_no=select_data)
    return int_plot