import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import gc
import csv
from scipy import stats
from interpret.chefer import *
from abc import ABC, abstractmethod
# assumes cuda availability ( will make robust later) and also need to add ability to batch in conformal prediction
from pytorch_grad_cam.metrics.road import ROADMostRelevantFirstAverage
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from road.imputations import *


# initializes an object that computes all calibration scores in PyTorch such that it's parallelizable
# note that interpreter has a model object.
class InterpretableCalibrationScore(ABC):
    def __init__(self, interpreter : STTransformerInterpreter):
        self.interpreter = interpreter
     
    # need this for the calibration scores
    # returns a N x 1 torch tensor of all scores per class_index
    @abstractmethod
    def cal_score(self, input, class_index):
        pass 

    # need this for the scoring function used in the prediction sets
    # must return a N x C torch tensor of all scores per class
    @abstractmethod
    def score(self, input):
        pass


class SoftMax(InterpretableCalibrationScore):
    def cal_score(self, input, class_index):
        # in principle they should have the same device and we detach to prevent memory leaks from autograd (I still don't entirely understand why that works)
        return self.interpreter.model(input.to(self.interpreter.device)).softmax(1).squeeze().gather(1, class_index.to(self.interpreter.device).unsqueeze(1)).squeeze().detach()
    def score(self, input):
        return self.interpreter.model(input.to(self.interpreter.device)).softmax(1).squeeze().detach()

# program cumulative softmax
class CumulativeSoftMax(InterpretableCalibrationScore):
    def cal_score(self, input, class_index):
        # get softmax scores across entire batch
        softmax_scores = self.interpreter.model(input.to(self.interpreter.device)).softmax(1).squeeze().detach()
        # sort them in descending order with respect to their own batches
        sorted_softmax_scores = torch.sort(softmax_scores, descending=True).values
        
        # get cumulative sum of sorted_softmax_scores
        # and then get the respective class index of cumulatively summed scores
        cal_scores = torch.cumsum(sorted_softmax_scores, dim=1).gather(1, class_index.to(self.interpreter.device).unsqueeze(1))

        # B x 1 tensor of calibration scores 
        return cal_scores.squeeze().detach()
    
    def score(self, input):
        # get softmax scores across entire batch
        softmax_scores = self.interpreter.model(input.to(self.interpreter.device)).softmax(1).detach()
        # sort them in descending order with respect to their own batches
        sorted_softmax_scores = torch.sort(softmax_scores, descending=True).values

        # get cumulative sum of sorted_softmax_scores
        scores = torch.cumsum(sorted_softmax_scores, dim=1)
        
        # B x C tensor of calibration scores
        return scores.squeeze().detach()

# GradCam++ Explainability Score Idea 
class SoftMaxDrop(InterpretableCalibrationScore):
    def cal_score(self, input, class_index):
        # get original softmax score
        default_softmax = 0
        with torch.no_grad():
            default_softmax = self.interpreter.model(input.to(self.interpreter.device)).softmax(1)
        if class_index == None:
            class_index = torch.argmax(default_softmax, axis=-1)
        # get cam inputted softmax score
        cam_input = self.interpreter.get_cam_on_image(input, class_index = class_index, method="linear")
        # np_img, cam_input = interpreter.visualize(input, class_index=class_index, show=False)
        # cam_input = cam_input * np_img
        
        interpreted_softmax = self.interpreter.model(cam_input.to(self.interpreter.device)).softmax(1).detach()
        # print(interpreted_softmax)
        # compute some differnence for the true class difference
        score = default_softmax - interpreted_softmax     # smaller difference == greater conformity
        # get the class_indices we want
        score = score.gather(1, class_index.to(self.interpreter.device).unsqueeze(1))
       
        # GREATER CONFIDENCE DROP == CAM IS MORE EXPLAINABLE == GREATER CONFORMITY OF HIGHLIGHTED REGION IN MODEL
        # CONFIDENCE DROP IS POSITIVE RIGHT NOW
        return score.detach()
    
    def score(self, input):
        og_softmax = 0
        with torch.no_grad():
            og_softmax = self.interpreter.model(input.to(self.interpreter.device)).softmax(1)
        interpreted_softmaxes_across_classes = torch.empty(input.size()[0], 0).to(self.interpreter.device)
        for c in range(og_softmax.size()[1]):
            cam = self.interpreter.get_cam_on_image(input, class_index = c, method='linear') # get cam-weighted image w/r to a possible class predictions
            interpreted_softmax = self.interpreter.model(cam.to(self.interpreter.device)).softmax(1)[:,c] # get corresponding softmax confidence to class
            # compute all softmax scores we care about w/r to each class!
            interpreted_softmaxes_across_classes = torch.cat((interpreted_softmaxes_across_classes, interpreted_softmax.unsqueeze(1)), dim=1)
        
        score = og_softmax - interpreted_softmaxes_across_classes
        # GREATER CONFIDENCE DROP == CAM IS MORE EXPLAINABLE == GREATER CONFORMITY OF HIGHLIGHTED REGION IN MODEL
        # CONFIDENCE DROP IS POSITIVE RIGHT NOW
        return score.detach()

# most relevant k pixels masked (removed) confidence drop from ROAR (without the retraining!)
class MORF(InterpretableCalibrationScore):
    def cal_score(self, input, class_index):
        input = input.to(self.interpreter.device) # should just run on the device of the interpreter class.
        # get original softmax score
        default_softmax = 0
        with torch.no_grad():
            default_softmax = self.interpreter.model(input.to(self.interpreter.device)).softmax(1)

        alphas = torch.tensor([.20, .40, .60, .80]) # hardcoded for now, q level?
        nSignals = input.size()[0]
        nPixels = input[0].view(-1).size()[0] # get total number of pixels
        # get cam mask
        cam = self.interpreter.get_cam(input, class_index = class_index, method="linear")
        # flatten came first
        cam = cam.view(nSignals,-1)
        avg_softmax_drop = torch.zeros(default_softmax.size()).to(self.interpreter.device)
        for alpha in alphas:
            q_level =  (torch.ceil((nPixels + 1) * (1 - alpha)) / nPixels).to(self.interpreter.device)
            kth_pixel_value = torch.quantile(cam, q_level, dim=1, interpolation="lower").to(self.interpreter.device)
            # mask all that is above the kth value
            masked_imgs = (cam < kth_pixel_value.unsqueeze(1)).view(input.size()) * input # unsqueeze because to match dims, need to unsqueeze
            # throw it back into the model and get a new matrix of softmax scores
            interpreted_softmax = self.interpreter.model(masked_imgs).softmax(1)
            avg_softmax_drop += (default_softmax - interpreted_softmax) / len(alphas)

        # get the class_indices we want
        avg_softmax_drop = avg_softmax_drop.gather(1, class_index.to(self.interpreter.device).unsqueeze(1))
       
        # GREATER CONFIDENCE DROP == CAM IS MORE EXPLAINABLE == GREATER CONFORMITY OF HIGHLIGHTED REGION IN MODEL
        # CONFIDENCE DROP IS POSITIVE RIGHT NOW
        # detach to reduce vram usage
        return avg_softmax_drop.detach()
    
    def score(self, input):
        input = input.to(self.interpreter.device)
        og_softmax = 0
        with torch.no_grad():
            og_softmax = self.interpreter.model(input.to(self.interpreter.device)).softmax(1)
        alphas = torch.tensor([.20, .40, .60, .80]) # hardcoded for now, q level?
        avg_softmax = torch.zeros(og_softmax.size()).to(self.interpreter.device)
        for c in range(og_softmax.size()[1]):
            nSignals = input.size()[0]
            nPixels = input[0].view(-1).size()[0] # get total number of pixels
            # get cam mask
            cam = self.interpreter.get_cam(input, class_index = c, method="linear")
            # flatten came first
            cam = cam.view(nSignals,-1)
            for alpha in alphas:
                q_level =  torch.ceil((nPixels + 1) * (1 - alpha)) / nPixels
                kth_pixel_value = torch.quantile(cam, q_level.to(self.interpreter.device), dim=1, interpolation="lower").to(self.interpreter.device)
                # mask all that is above the kth value, we use < s.t big values are set to 0
                masked_imgs = (cam < kth_pixel_value.unsqueeze(1)).view(input.size()) * input # unsqueeze because to match dims, need to unsqueeze
                # throw it back into the model and get a new matrix of softmax scores
                interpreted_softmax = self.interpreter.model(masked_imgs).softmax(1)[:,c]
            
                avg_softmax[:,c] += (interpreted_softmax) / len(alphas) # should a 1D vector of softmaxes across each class

        # GREATER CONFIDENCE DROP == CAM IS MORE EXPLAINABLE == GREATER CONFORMITY OF HIGHLIGHTED REGION IN MODEL
        # CONFIDENCE DROP IS POSITIVE RIGHT NOW

        score = og_softmax - avg_softmax
        return score.detach()



class ROAD(InterpretableCalibrationScore):
    def cal_score(self, input, class_index):
        # Compute Original Softmax Scores for Batch
        default_softmax = 0
        with torch.no_grad():
            default_softmax = self.interpreter.model(input.to(self.interpreter.device)).softmax(1)
        imputer = NoisyLinearImputer()
        alphas = torch.tensor([.20, .40, .60, .80]) # hardcoded for now, q level?
        nSignals = input.size()[0]
        nPixels = input[0].view(-1).size()[0] # get total number of pixels
        # get cam mask
        cam = self.interpreter.get_cam(input, class_index = class_index, method="linear")
        # flatten came first
        cam = cam.view(nSignals,-1)
        avg_softmax_drop = torch.zeros(default_softmax.size()).to(self.interpreter.device)
        for alpha in alphas:
            q_level =  (torch.ceil((nPixels + 1) * (1 - alpha)) / nPixels).to(self.interpreter.device)
            kth_pixel_value = torch.quantile(cam, q_level, dim=1, interpolation="lower").to(self.interpreter.device)
            # mask all that is above the kth value
            mask= (cam < kth_pixel_value.unsqueeze(1)).view(input.size())  # unsqueeze because to match dims, need to unsqueeze
            # Use NoisyLinearImputer to Impute the Masked Region On the Original Image
            # unsqueeze for color channel lol
            imputed_imgs = imputer.batched_call(input.unsqueeze(1), mask)
            imputed_imgs = imputed_imgs.squeeze().to(self.interpreter.device) # squeeze it back into normal shape
            # throw it back into the model and get a new matrix of softmax scores
            interpreted_softmax = self.interpreter.model(imputed_imgs).softmax(1)
            avg_softmax_drop += (default_softmax - interpreted_softmax) / len(alphas)

        
        avg_softmax_drop = avg_softmax_drop.gather(1, class_index.to(self.interpreter.device).unsqueeze(1))
        return avg_softmax_drop.detach()


    def score(self, input):
        input = input.to(self.interpreter.device)
        og_softmax = 0
        with torch.no_grad():
            og_softmax = self.interpreter.model(input.to(self.interpreter.device)).softmax(1)
        alphas = torch.tensor([.20, .40, .60, .80]) # hardcoded for now, q level?
        avg_softmax = torch.zeros(og_softmax.size()).to(self.interpreter.device)
        imputer = NoisyLinearImputer()
        for c in range(og_softmax.size()[1]):
            nSignals = input.size()[0]
            nPixels = input[0].view(-1).size()[0] # get total number of pixels
            # get cam mask
            cam = self.interpreter.get_cam(input, class_index = c, method="linear")
            # flatten came first
            cam = cam.view(nSignals,-1)
            for alpha in alphas:
                q_level =  torch.ceil((nPixels + 1) * (1 - alpha)) / nPixels
                kth_pixel_value = torch.quantile(cam, q_level.to(self.interpreter.device), dim=1, interpolation="lower").to(self.interpreter.device)
                # mask all that is above the kth value
                mask = (cam < kth_pixel_value.unsqueeze(1)).view(input.size()) # unsqueeze because to match dims, need to unsqueeze
                imputed_imgs = imputer.batched_call(input.unsqueeze(1) ,mask)
                imputed_imgs = imputed_imgs.squeeze()
                # throw it back into the model and get a new matrix of softmax scores
                interpreted_softmax = self.interpreter.model(imputed_imgs.view(input.size())).softmax(1)[:,c]
            
                avg_softmax[:,c] += (interpreted_softmax) / len(alphas) # should a 1D vector of softmaxes across each class

        # GREATER CONFIDENCE DROP == CAM IS MORE EXPLAINABLE == GREATER CONFORMITY OF HIGHLIGHTED REGION IN MODEL
        # CONFIDENCE DROP IS POSITIVE RIGHT NOW

        score = og_softmax - avg_softmax
        return score.detach()