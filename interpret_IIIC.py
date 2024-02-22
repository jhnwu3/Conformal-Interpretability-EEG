import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import skimage

from interpret.chefer import *
from models.st_transformer import *
from models.pytorch_lightning import *
from data import *



class_labels = {
    0 : 'Seizure', 1 : 'LPD', 2 : 'GPD', 3 : 'LRDA', 4: 'GRDA', 5 :'Other'
}# load the dataset

train_loader, test_loader, val_loader, cal_loader = prepare_IIIC_cal_dataloader(drop_last=True)
train = train_loader.dataset
test = test_loader.dataset

# visualize one example
index = 3120
signal, label = test[index]
signal = signal.unsqueeze(0) # assume batch dim
plt.figure(figsize=(50,50))
# do first 200 to make figure less hard to see.
plt.imshow(signal[:,:,:].squeeze().numpy(), cmap='gray')
print(class_labels[label])
print(signal.size())

# load the model
# model = STTransformer(depth=4, n_classes=6, channel_length=1000, dropout=0.5)
 # my model hyperparameters
emb_size = 256
depth = 6 
dropout = 0.5
num_heads = 8
patch_kernel_length = 11  # cqi = 15 - UNUSED
stride = 11  # cqi = 8 - UNUSED

model = STTransformer(emb_size=emb_size, 
                                depth=depth,
                                n_classes=6, 
                                channel_length=2000,
                                dropout=dropout, 
                                num_heads=num_heads,
                                kernel_size=11, 
                                stride=11,
                                kernel_size2=11,
                                stride2=11)

model.load_state_dict(torch.load("saved_weights/IIIC_st_transformer_conformal_c11s11c5s5.pt"))
# get visualization
model(signal).size()

print(model.channel_attention.get_attn_map().size()) # 1. Can we ignore channel attention? Probably no, because we want to highlight those areas highlighted by channel attention too
print(model.transformer.transformer_blocks[0].mhattn.get_attn_map().size()) # 2. Can we do this first? Yes
interpreter = STTransformerInterpreter(model=model)

test_iter = iter(test_loader)
signal_batch, signal_labels = next(test_iter)
sequence, attribution = interpreter.visualize(signal_batch, class_index = signal_labels, save_path=None, figsize=(100,16), method="linear")
sequence, attribution = interpreter.visualize(signal, class_index = label, save_path=None, figsize=(100,16), method="linear")
cam = interpreter.get_cam(signal, class_index = label)
cam = interpreter.get_cam(signal_batch, class_index=signal_labels)
cam_img = interpreter.get_cam_on_image(signal, class_index = label)
cam_img = interpreter.get_cam_on_image(signal_batch, class_index=signal_labels)

# plt.imshow(vis)
interpreter.get_top_classes(signal, class_labels=class_labels)
print("True Label:", class_labels[label])