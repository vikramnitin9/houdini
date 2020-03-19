import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision
import torchvision.models as models
import torchvision.datasets as datasets

from torch.utils.data import Dataset, DataLoader

import pickle
import random
import time
import os
import copy
import time
import math

from PIL import Image

import argparse

from pprint import pprint
from collections import defaultdict as ddict

import logging, uuid, sys

import numbers

class RemapClasses(object):
    def __init__(self, class_remapping):
        self.class_remapping = class_remapping

    def __call__(self, target):
        assert isinstance(target, torch.Tensor)

        for i in self.class_remapping:
            target[target == i] = self.class_remapping[i]

        return target

def get_logger(name, log_dir):
	logger = logging.getLogger(name)
	logFormatter = logging.Formatter("%(asctime)s - %(name)s - [%(levelname)s] - %(message)s")

	fileHandler = logging.FileHandler("{0}/{1}.log".format(log_dir, name.replace('/', '-')))
	fileHandler.setFormatter(logFormatter)
	logger.addHandler(fileHandler)

	consoleHandler = logging.StreamHandler(sys.stdout)
	consoleHandler.setFormatter(logFormatter)
	logger.addHandler(consoleHandler)

	logger.setLevel(logging.INFO)

	return logger

def set_gpu(gpus):
	os.environ["CUDA_DEVICE_ORDER"]    = "PCI_BUS_ID"
	os.environ["CUDA_VISIBLE_DEVICES"] = gpus

def fast_hist(pred, label, n):
    k = (label >= 0) & (label < n)
    return np.bincount(
        n * label[k].astype(int) + pred[k], minlength=n ** 2).reshape(n, n)

def per_class_iu(hist):
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))

def per_image_fast_hist(pred, label, n):
    k           = (label >= 0) & (label < n)
    label[~k]   = n
    temp        = np.apply_along_axis(np.bincount, 1, n * label.astype(int) + pred, minlength=(n**2) + n)
    temp        = temp[:, : n**2]

    return temp.reshape(pred.shape[0], n, n)

def per_image_per_class_iu(hist):
    diag = np.diagonal(hist, axis1=1, axis2=2)

    return diag / (hist.sum(2) + hist.sum(1) - diag + 1e-10)

def per_image_iou(Y_pred, Y):
    num_classes         = Y_pred.shape[1]

    class_prediction    = torch.argmax(Y_pred, dim=1)
    class_prediction    = class_prediction.cpu().detach().numpy().reshape(class_prediction.shape[0], -1)
    target_seg          = Y.cpu().detach().numpy().reshape(Y.shape[0], -1)

    hist                = per_image_fast_hist(class_prediction, target_seg, num_classes)
    ious                = per_image_per_class_iu(hist) * 100
    mIoU                = np.nanmean(ious, axis=1)
    mIoU                = torch.Tensor(mIoU).to(Y_pred.device)

    return mIoU