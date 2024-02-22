import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import pytorch_lightning as pl
from models import *
from interpret.chefer import *
from models.st_transformer import *
from pyhealth.metrics import multiclass_metrics_fn



# jerry rigged Chaoqi's implementaiton of training with Python lightning
class LitModel(pl.LightningModule):
    def __init__(self,model, lr = 0.001, w_decay=1e-5):
        super().__init__()
        self.model = model
        self.lr = lr 
        self.w_decay = w_decay
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def training_step(self, batch, batch_idx):
        X, y = batch
        prod = self.model(X)
        loss = nn.CrossEntropyLoss()(prod, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        X, y = batch
        with torch.no_grad():
            convScore = self.model(X)
            step_result = convScore.cpu().numpy()
            step_gt = y.cpu().numpy()
        
        self.validation_step_outputs.append((step_result, step_gt))
        return step_result, step_gt

    def on_validation_epoch_end(self):
        result = []
        gt = np.array([])
        for out in self.validation_step_outputs:
            result.append(out[0])
            gt = np.append(gt, out[1])

        result = np.concatenate(result, axis=0)
        result = multiclass_metrics_fn(
            gt, result, metrics=["accuracy", "cohen_kappa", "f1_weighted"]
        )
        self.log("val_acc", result["accuracy"], sync_dist=True)
        self.log("val_cohen", result["cohen_kappa"], sync_dist=True)
        self.log("val_f1", result["f1_weighted"], sync_dist=True)
        print(result)

    def test_step(self, batch, batch_idx):
        X, y = batch
        with torch.no_grad():
            convScore = self.model(X)
            step_result = convScore.cpu().numpy()
            step_gt = y.cpu().numpy()
            self.test_step_outputs.append((step_result, step_gt))
        return step_result, step_gt

    def on_test_epoch_end(self):
        result = []
        gt = np.array([])
        for out in self.test_step_outputs:
            result.append(out[0])
            gt = np.append(gt, out[1])

        result = np.concatenate(result, axis=0)
        result = multiclass_metrics_fn(
            gt, result, metrics=["accuracy", "cohen_kappa", "f1_weighted"]
        )
        self.log("test_acc", result["accuracy"], sync_dist=True)
        self.log("test_cohen", result["cohen_kappa"], sync_dist=True)
        self.log("test_f1", result["f1_weighted"], sync_dist=True)

        return result

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.w_decay,
        )

        return [optimizer]  # , [scheduler]
