import numpy as np
import torch
import math
from .base_model import BaseModel
from . import networks
from .patchnce import PatchNCELoss
import util.util as util
import models.voxelmorph.torchvoxelmorph as vxm
from models.voxelmorph.torchvoxelmorph.layers import SpatialTransformer
from PIL import Image
from torchvision.transforms import Compose, CenterCrop, ToTensor, Normalize
import torch.nn.functional as F
from util.losses import NCC_Loss, NMI_Loss

def open_image_to_torch(path, size):
    t = Image.open(path)
    transforms = []
    transforms.append(CenterCrop(size))
    transforms.append(ToTensor())
    transforms.append(Normalize(mean=[0.5], std=[0.5]))
    transforms = Compose(transforms)
    t = transforms(t)
    t = t.unsqueeze(0)
    return t

def smooothing_loss(y_pred):
    dy = torch.abs(y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :])
    dx = torch.abs(y_pred[:, :, :, 1:] - y_pred[:, :, :, :-1])

    dx = torch.mul(dx, dx)
    dy = torch.mul(dy, dy)
    d = torch.mean(dx) + torch.mean(dy)
    return d/2.0

class REGISTRATIONModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """  Configures options specific for CUT model
        """
        parser.add_argument('--CUT_mode', type=str, default="CUT", choices='(CUT, cut, FastCUT, fastcut)')

        parser.add_argument('--lambda_GAN', type=float, default=0.0, help='weight for GAN loss：GAN(G(X))')
        parser.add_argument('--lambda_NCE', type=float, default=0.25, help='weight for NCE loss: NCE(G(X), X)')
        parser.add_argument('--nce_T', type=float, default=0.07, help='temperature for NCE loss')
        parser.add_argument('--num_patches', type=int, default=256, help='number of patches per layer')
        parser.add_argument('--flip_equivariance',
                            type=util.str2bool, nargs='?', const=True, default=False,
                            help="Enforce flip-equivariance as additional regularization. It's used by FastCUT, but not CUT")

        parser.set_defaults(pool_size=0)  # no image pooling

        opt, _ = parser.parse_known_args()
        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        self.loss_names = ['G', 'R', 'smooth', 'MMR']
        self.visual_names = ['real_A', 'fake_B', 'real_B', 'dvf', 'registered', 'regA']
        if self.isTrain:
            self.model_names = ['G', 'R']
        else:  
            self.model_names = ['G', 'R']

       
        self.netG = networks.define_G(opt.inn_input_nc, opt.inn_output_nc, opt.ngf, opt.netG, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, opt.no_antialias_up, self.gpu_ids, opt)  # ResNet/..
        
        nb_features = [
            [16, 32, 32, 64, 64, 64],  # encoder
            [64, 64, 64, 32, 32, 32, 16]  # decoder
        ]
        vol_shape = (opt.crop_size, opt.crop_size)
        self.netR = vxm.networks.VxmDense(vol_shape, nb_features, int_steps=7, bidir=True).cuda() #VoxelMorph
        self.netR.train()
        self.spatialTransformer = SpatialTransformer(vol_shape).cuda()


        if self.isTrain:
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionIdt = torch.nn.L1Loss().to(self.device)
            self.criterionNMI = NMI_Loss([0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95],device=self.device)

            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizer_R = torch.optim.Adam(self.netR.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizer_F = None
            
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_R)  

    def data_dependent_initialize(self, data):
       
        self.set_input(data)
        bs_per_gpu = self.real_A.size(0) // max(len(self.opt.gpu_ids), 1)
        self.real_A = self.real_A[:bs_per_gpu]
        self.real_B = self.real_B[:bs_per_gpu]
        self.forward()                     # compute fake images: G(A)
    
    def optimize_parameters(self):
        # forward
        self.forward()

     
        y_output = self.netR(self.real_A, self.real_B) # y_output: deformation field
        y_pred = [self.spatialTransformer(self.fake_B, y_output[2]),y_output[2]]
       
        self.registered = y_pred[0] # R(T(A))
        self.regA = y_output[0]
      
        self.dvf = y_pred[1]

        self.optimizer_G.zero_grad()
        self.optimizer_R.zero_grad()
      
        self.loss_G = self.compute_INN_loss() * self.opt.w_G 

        # R loss
        mask = (self.real_B>-0.95) + (y_pred[0] >-0.95)
        mask2 = (self.idt_B>-0.95) + (y_pred[0] >-0.95)
        self.loss_MMR = (-torch.log(self.opt.w_barrier - self.criterionNMI(y_output[0], self.real_B)) )* self.opt.w_MMR
        self.loss_R = self.calculate_L1_loss(y_pred[0],self.real_B)*1.0 + self.calculate_L1_loss(self.fake_A, y_output[0])*1.0 #+ self.loss_local*1.0
        self.loss_smooth = smooothing_loss(y_pred[1]) * self.opt.w_smooth #10.0 # 2.0 #0.20

        all_G_loss = self.loss_R + self.loss_G + self.loss_MMR  + self.loss_smooth 
        all_G_loss.backward()

        if self.opt.optz_G:
            self.optimizer_G.step()
        if self.opt.optz_R:
   
    def set_input(self, input):
       
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.realAA = torch.cat((self.real_A, self.real_A), dim=1) # torch.Size([2, 2, 256, 256])
        self.fake = self.netG(self.realAA)  
        self.fake_B = self.fake[:,self.real_A.size(1):] #torch.Size([2, 1, 256, 256])
        self.reconAA = self.netG(self.fake, rev=True)

        ## Reversed INN
        self.realBB = torch.cat((self.real_B, self.real_B), dim=1) # torch.Size([2, 2, 256, 256])
        self.fake_rev = self.netG(self.realBB, rev=True)  
        self.fake_A = self.fake_rev[:,self.real_A.size(1):] #torch.Size([2, 1, 256, 256])
        self.reconBB = self.netG(self.fake_rev)


        if not self.opt.isTrain:
            y_output = self.netR(self.real_A, self.real_B) # y_output: deformation field
            y_pred = [self.spatialTransformer(self.fake_B, y_output[2]),y_output[2]]
            self.registered = y_pred[0]
            self.regA = y_output[0]
            self.dvf = y_pred[1]

  
    def compute_G_loss(self):
        fake = self.fake_B
        if self.opt.lambda_GAN > 0.0:
            pred_fake = self.netD(fake)
            self.loss_G_GAN = self.criterionGAN(pred_fake, True).mean() * self.opt.lambda_GAN
        else:
            self.loss_G_GAN = 0.0
        self.loss_G = self.loss_G_GAN
        return self.loss_G


    def compute_INN_loss(self):
        """Calculate loss for the INN"""
        self.loss_INN_AA = self.calculate_L1_loss(self.realAA, self.reconAA)
        self.loss_INN_BB = self.calculate_L1_loss(self.realBB, self.reconBB)
        self.loss_INN_L1 = self.loss_INN_AA + self.loss_INN_BB
        return self.loss_INN_L1


    def calculate_L1_loss(self, src, tgt, mask=None):
        diff = torch.abs(src - tgt)
        if mask is None:
            return torch.mean(diff)
        elif torch.sum(mask) == 0:
            return torch.tensor(0)
        else:
            norm_factor = 1 / (torch.sum(mask))
            return norm_factor * torch.sum(diff * mask)
 