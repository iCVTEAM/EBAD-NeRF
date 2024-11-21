import torch
import math
import numpy as np
import cv2
import random
import os

def lin_log(x, threshold=20):
    """
    linear mapping + logarithmic mapping.
    :param x: float or ndarray the input linear value in range 0-255
    :param threshold: float threshold 0-255 the threshold for transisition from linear to log mapping
    """
    # converting x into np.float32.
    if x.dtype is not torch.float64:
        x = x.double()
    f = (1./threshold) * math.log(threshold)
    y = torch.where(x <= threshold, x*f, torch.log(x))
    #rounding = 1e8
    #y = torch.round(y*rounding)/rounding
    return y.float()

def event_loss_call_blender(all_rgb, event_data, rgb2grey, bin_num):
    loss = []
    for its in range(bin_num):
        start = its
        end = its + 1
        thres = (torch.log(torch.mv(all_rgb[end], rgb2grey) * 255) - torch.log(torch.mv(all_rgb[start], rgb2grey) * 255)) / 0.25
        event_cur = event_data[start]
        loss.append(torch.mean((thres - event_cur) ** 2))
    event_loss = torch.mean(torch.stack(loss, dim=0), dim=0)
    return event_loss

def event_loss_call_davis(all_rgb, event_data, rgb2grey, bin_num):
    loss = []
    for its in range(bin_num):
        start = its
        end = its + 1
        thres = (torch.log(torch.mv(all_rgb[end], rgb2grey) * 255) - torch.log(torch.mv(all_rgb[start], rgb2grey) * 255)) / 0.35
        event_cur = event_data[start]
        loss.append(torch.mean((thres - event_cur) ** 2))
    event_loss = torch.mean(torch.stack(loss, dim=0), dim=0)
    return event_loss


