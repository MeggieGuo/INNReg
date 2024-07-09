import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
from tqdm import tqdm
import json
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.autograd import Variable
from util import metric
from torch.utils.tensorboard import SummaryWriter
import imageio
from skimage import img_as_ubyte

try:
    import wandb
except ImportError:
    print('Warning: wandb package cannot be found. The option "--use_wandb" will result in error.')


def print_metric(metric_name, metrics):
    a1 = [entry[metric_name] for entry in metrics]
    print(metric_name, np.round(np.mean(a1),5), np.round(np.std(a1),5))


class SpatialTransformer(nn.Module):
    def __init__(self, size, mode='bilinear'):
        super(SpatialTransformer, self).__init__()
        # Create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)  # y, x, z
        grid = torch.unsqueeze(grid, 0)  # add batch
        grid = grid.type(torch.FloatTensor)
        self.register_buffer('grid', grid)

        self.mode = mode

    def forward(self, src, flow):
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        # Need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return F.grid_sample(src, new_locs, mode=self.mode)



def test(opt):
    opt.num_threads = 0   # test code only supports num_threads = 0
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers

    metrics=[]

    STN = SpatialTransformer([opt.load_size,opt.load_size]).cuda()
    STN.eval()

    if opt.eval:
        model.eval()
    for i, data in enumerate(tqdm(dataset)):
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
       
        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference
        visuals = model.get_current_visuals()  # get image results
        
        moving = visuals['real_A']/2+0.5
        TransA = visuals['fake_B']/2+0.5
        ReconA = visuals['registered']/2+0.5
        fixed = visuals['real_B']/2+0.5
        warped = visuals['regA']/2+0.5
        dvf = visuals['dvf']#.permute(0, 3, 1, 2)

        seg_moving = Variable(data['moving_seg'].cuda())/2+0.5
        seg_fixed = Variable(data['fixed_seg'].cuda())/2+0.5
        gt = Variable(data['moving_gt'].cuda())/2+0.5
        dvf_gt = data['dvf_gt']
        img_path = data['B_paths']

    
        name = os.path.basename(img_path[0]).split('.')[0] 

        pred_seg = STN(seg_moving, dvf) #torch.Size([16, 1, 192, 192])

        save_path = os.path.join('checkpoints/Results_CycleBased', pro_name, '{}'.format(name))
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        print(save_path)
        imageio.imwrite(os.path.join(save_path, 'moving.png'), img_as_ubyte(moving[0,0,:,:].detach().cpu().numpy()))
        imageio.imwrite(os.path.join(save_path, 'fixed.png'), img_as_ubyte(fixed[0,0,:,:].detach().cpu().numpy()))
        imageio.imwrite(os.path.join(save_path, 'warped.png'), img_as_ubyte(warped[0,0,:,:].detach().cpu().numpy()))
        imageio.imwrite(os.path.join(save_path, 'TransA.png'), TransA[0,0,:,:].detach().cpu().numpy())
        imageio.imwrite(os.path.join(save_path, 'ReconA.png'), ReconA[0,0,:,:].detach().cpu().numpy())
        imageio.imwrite(os.path.join(save_path, 'gt.png'), img_as_ubyte(gt[0,0,:,:].detach().cpu().numpy()))
        imageio.imwrite(os.path.join(save_path, 'seg_moving.png'), img_as_ubyte(seg_moving[0,0,:,:].detach().cpu().numpy()))
        imageio.imwrite(os.path.join(save_path, 'seg_fixed.png'), img_as_ubyte(seg_fixed[0,0,:,:].detach().cpu().numpy()))
        imageio.imwrite(os.path.join(save_path, 'seg_warped.png'), img_as_ubyte(pred_seg[0,0,:,:].detach().cpu().numpy()))

      

if __name__ == '__main__':

    visual_path = '/bask/projects/d/duanj-ai-imaging/mxg/checkpoints/VisualTB/a_InnReg'
    save_path = 'Checkpoint'
    pro_name = '***' 
    
    opt = TestOptions().parse()  # get training options
    with open(os.path.join(save_path, pro_name, 'test_setting.json'), 'r') as f:
        opt.__dict__.update(json.load(f))
    for arg in vars(opt):
        print(format(arg, '<20'), format(str(getattr(opt, arg)), '<'))   # str, arg_type

    opt.name = pro_name
    # exit()
    writer = SummaryWriter(visual_path)
    test(opt)