import time
import torch
import os
import json
import numpy as np
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib.pyplot as plt



def train(opt):
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)    # get the number of images in the dataset.
    model = create_model(opt)      # create a model given opt.model and other options
    print('The number of training images = %d' % dataset_size)

    total_iters = 0                # the total number of training iterations

    optimize_time = 0.1

    times = []
    for epoch in tqdm(range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1)):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch

        if epoch >= opt.epoch_1 and epoch % 2 == 0 :
            opt.optz_G = True            
            opt.optz_R = False
        # if epoch >= opt.epoch_2:
        elif epoch >= opt.epoch_1 and epoch % 2 == 1 :
            opt.optz_R = True
            opt.optz_G = False
        print(f'Current optimizier strategy: optimizer_G {opt.optz_G}, optimizer_R {opt.optz_R}')

        dataset.set_epoch(epoch)
        for i, data in enumerate(tqdm(dataset)):  # inner loop within one epoch
     
            epoch_iter += opt.batch_size
            if epoch == opt.epoch_count and i == 0:
                model.data_dependent_initialize(data)
                model.setup(opt)               # regular setup: load and print networks; create schedulers

            model.set_input(data)  # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

            total_iters += 1
         
            if total_iters % opt.display_freq == 0:   # display images on visdom and save images to a HTML file
                model.compute_visuals()
                visual_dict = model.get_current_visuals() 
                
                writer.add_images(pro_name+"/train_realA", visual_dict['real_A']/2+0.5, global_step = epoch, dataformats='NCHW')
                writer.add_images(pro_name+"/train_realB", visual_dict['real_B']/2+0.5, global_step = epoch, dataformats='NCHW')
                writer.add_images(pro_name+"/train_TrAB", visual_dict['fake_B']/2+0.5, global_step = epoch, dataformats='NCHW') #T(A)
                writer.add_images(pro_name+"/train_regA", visual_dict['regA']/2+0.5, global_step = epoch, dataformats='NCHW')
                writer.add_images(pro_name+"/train_registeredA", visual_dict['registered']/2+0.5, global_step = epoch, dataformats='NCHW')
             
            if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                writer.add_scalars(pro_name+'/loss', losses, total_iters)

        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('new')
        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))

        model.update_learning_rate() # update learning rates at the end of every epoch.

    writer.close()

if __name__ == '__main__':

    visual_path = '/bask/projects/d/duanj-ai-imaging/mxg/checkpoints/VisualTB/a_InnReg'
    save_path = 'Checkpoint'
    pro_name = '***' 

    if not os.path.exists(visual_path):
        os.makedirs(visual_path)
    
    opt = TrainOptions().parse()  # get training options
    with open(os.path.join(save_path, pro_name, 'setting.json'), 'r') as f:
        opt.__dict__.update(json.load(f))

    opt.name = pro_name
    opt.checkpoints_dir = save_path
    
    for arg in vars(opt):
        print(format(arg, '<20'), format(str(getattr(opt, arg)), '<'))   # str, arg_type
    
    writer = SummaryWriter(visual_path)
    train(opt)