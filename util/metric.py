import os,glob,imageio

import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
import os.path as osp
import torch.nn.functional as F
from torch.nn import MSELoss
from torchvision import transforms
# from data_affine import ImageFolder
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
import matplotlib.pyplot as plt
# import kornia
import torch
from scipy.spatial.distance import directed_hausdorff
import numpy as np
import cv2
from skimage.metrics import structural_similarity as compare_ssim


def MSE(img1, img2):
    img1 = img1.detach().cpu().numpy()
    img2 = img2.detach().cpu().numpy()
    return np.mean((img1 - img2) ** 2)

def NCC(img1,img2):
    img1 = img1.detach().cpu().numpy()
    img2 = img2.detach().cpu().numpy()
    r = np.mean(np.multiply((img1-np.mean(img1)),(img2-np.mean(img2))))/(np.std(img1)*np.std(img2))
    return r

def Dice(img1, gt):
    img1 = img1.detach().cpu().numpy()
    gt = gt.detach().cpu().numpy()

    gt = np.where(gt > 0, 1, gt)
    img1 = np.where(img1 > 0, 1, img1)

    dice = 2 *(img1 * gt).sum() / (img1.sum() + gt.sum())
    return dice

def Dice3(img1,img2):
    _,_,H,W= img1.shape
    fenmu1 = 0
    fenmu2 = 0
    fenzi = 0
    for i in range(H):
        for j in range(W):
            if (img1[0,0,i,j] != 0):
                fenmu1 += 1
            if (img2[0,0,i,j] != 0):
                fenmu2 += 1
            if (img1[0,0,i,j] != 0)&(img2[0,0,i,j] != 0):
                fenzi += 1

    dice = 2*fenzi/(fenmu1+fenmu2)
    return dice

def HD95(image1, image2):
    # Assuming you have two tensor images, image1 and image2
    # Each image should be a tensor with shape (H, W), representing grayscale images

    # Convert the tensor images to NumPy arrays
    contours1 = image1[0,:,:,:].detach().cpu().numpy()
    contours2 = image2[0,:,:,:].detach().cpu().numpy()

    # # Perform image processing to extract contours (sets of points) from the images
    # contours1, _ = cv2.findContours(image1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # contours2, _ = cv2.findContours(image2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Calculate the directed Hausdorff distance from contours1 to contours2
    directed_hd1 = directed_hausdorff(contours1[0], contours2[0])

    # Calculate the directed Hausdorff distance from contours2 to contours1
    directed_hd2 = directed_hausdorff(contours2[0], contours1[0])

    # The 95th percentile Hausdorff distance is the maximum of the two directed distances
    percentile = 95  # You can change this to a different percentile if needed
    hausdorff_distance_95 = np.percentile([directed_hd1, directed_hd2], percentile)

    # # Print the result
    # print(f"The {percentile}th percentile Hausdorff distance is: {hausdorff_distance_95}")
    return hausdorff_distance_95

def SSIM(image1, image2):

    image1 = image1[0,0,:,:].detach().cpu().numpy()
    image2 = image2[0,0,:,:].detach().cpu().numpy()

    # Convert images to 8-bit integers (0-255 range) if not already
    image1 = np.uint8(image1 * 255)
    image2 = np.uint8(image2 * 255)

    # Calculate SSIM
    ssim = compare_ssim(image1, image2)

    return ssim

def PSNR(image1, image2):

    image1 = image1[0,0,:,:].detach().cpu().numpy()
    image2 = image2[0,0,:,:].detach().cpu().numpy()
    
    # Convert images to 8-bit integers (0-255 range) if not already
    image1 = np.uint8(image1 * 255)
    image2 = np.uint8(image2 * 255)

    # Calculate PSNR
    mse = np.mean((image1 - image2) ** 2)
    psnr = 10 * np.log10((255**2) / mse)

    return psnr


def gradient_loss(s, penalty='l2'):
    # dy = torch.abs(s[:, :, 1:, :, :] - s[:, :, :-1, :, :])
    dx = torch.abs(s[:, :, 1:, :] - s[:, :, :-1, :])
    dz = torch.abs(s[:, :, :, 1:] - s[:, :, :, :-1])

    if (penalty == 'l2'):
        # dy = dy * dy
        dx = dx * dx
        dz = dz * dz

    d = torch.mean(dx)  + torch.mean(dz)
    return (d / 2.0).detach().cpu().numpy()
