import torch
from .base_model import BaseModel
from collections import OrderedDict
from . import networks
import torch.nn.functional as F
from torch import nn, cuda
from torch.autograd import Variable
import time
import numpy as np
from util import util
import os
import itertools
from torchstat import stat
import time


class ObAdaINModel(BaseModel):
    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.opt = opt
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G', 'c', 's', 'rec', 'class', 'tv']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        if self.isTrain:
            self.visual_names = ['comp', 'comp_mask_vis', 'style', 'style_mask_vis', 'final_output', 'coarse_output', 'att_mask']
        else:
            self.visual_names = ['style', 'comp_mask_vis', 'comp', 'final_output']

        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        self.model_names = ['G']
        # define networks
        if opt.is_skip:
            self.netDecoder = networks.decoder_cat
        else:
            self.netDecoder = networks.decoder
        self.netvgg = networks.vgg
        self.netvgg.load_state_dict(torch.load(opt.vgg))
        self.netvgg = nn.Sequential(*list(self.netvgg.children())[:31])
        self.netG = networks.ArtoNet(self.netvgg, self.netDecoder, is_matting=opt.is_matting)
        #self.netG = networks.VGGNet(self.netvgg, self.netDecoder)
        if len(self.gpu_ids) > 0:
            assert(torch.cuda.is_available())
            self.netG.to(self.gpu_ids[0])
            self.netG = torch.nn.DataParallel(self.netG, self.gpu_ids)


        if self.isTrain:
            # losses are calculated in network.
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(filter(lambda p: p.requires_grad, self.netG.parameters()), lr=opt.lr*opt.g_lr_ratio, betas=(opt.beta1, opt.beta2))
            self.optimizers.append(self.optimizer_G)


    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.
        """
        self.content = input['content'].to(self.device)
        self.comp = input['comp'].to(self.device)
        self.comp_mask_vis = input['comp_mask'].to(self.device)
        self.comp_mask = self.comp_mask_vis / 2 + 0.5
        self.style = input['style'].to(self.device)
        self.style_mask_vis = input['style_mask'].to(self.device)
        self.style_mask = self.style_mask_vis / 2 + 0.5

        if self.isTrain:
            self.comp_patch_mask = input['comp_patch_mask'].to(self.device)

    def forward(self):
        if self.isTrain:
            self.coarse_output, self.final_output, self.att_mask, self.loss_c, self.loss_s, self.loss_rec, self.loss_class, self.loss_tv, self.loss_mask, self.comp_content_feat, self.style_content_feat = \
                    self.netG(self.comp, self.style, self.comp_mask, self.style_mask, self.isTrain)
        else:
            self.coarse_output, self.final_output, self.att_mask, self.comp_content_feat, self.style_content_feat = self.netG(self.comp, self.style, self.comp_mask, self.style_mask)


    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""

        self.loss_G = self.opt.lambda_c * self.loss_c + self.opt.lambda_s * self.loss_s + self.opt.lambda_rec * self.loss_rec + \
                      self.opt.lambda_class * self.loss_class + self.opt.lambda_tv * self.loss_tv
        print('loss: c {},s {},rec {},class {},tv {}'.format(self.loss_c.item(),self.loss_s.item(),self.loss_rec.item(),self.loss_class.item(),self.loss_tv.item()))
        self.loss_G.backward(retain_graph=True)

    def optimize_parameters(self):
        self.forward()

        self.optimizer_G.zero_grad()  # set G's gradients to zero
        self.backward_G()  # calculate graidents for G
        self.optimizer_G.step()  # udpate G's weights
    

    def get_current_visuals(self):
        t= time.time()
        num = min(8, self.style.size(0))
        visual_ret = OrderedDict()
        all =[]

        for i in range(0,num):
            row=[]
            for name in self.visual_names:
                if isinstance(name, str):
                    if hasattr(self,name):
                        im = util.tensor2im(getattr(self, name).data[i:i+1,:,:,:])
                        row.append(im)
            row=tuple(row)
            all.append(np.hstack(row))
        all = tuple(all)

        allim = np.vstack(all)
        return OrderedDict([(self.opt.name,allim)])
    

    

