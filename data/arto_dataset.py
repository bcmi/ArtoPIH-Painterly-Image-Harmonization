import os.path
import torch
import random
import torchvision.transforms.functional as tf
from torchvision.transforms import Resize 
from data.base_dataset import BaseDataset, get_transform
import PIL

from PIL import Image
import numpy as np
#import torchvision.transforms as transforms
from pathlib import Path

# ImageFile.LOAD_TRUNCATED_IMAGES = True
from PIL import Image, ImageFile
Image.MAX_IMAGE_PIXELS = None  # Disable DecompressionBombError
ImageFile.LOAD_TRUNCATED_IMAGES = True
import time
import json

import cv2
random.seed(1)

def mask_bboxregion(mask):
    mask = np.array(mask)
    w,h = np.shape(mask)[:2]
    valid_index = np.argwhere(mask==255) # [length,2]
    if np.shape(valid_index)[0] < 1:
        x_left = 0
        x_right = 0
        y_bottom = 0
        y_top = 0
    else:
        x_left = np.min(valid_index[:,0])
        x_right = np.max(valid_index[:,0])
        y_bottom = np.min(valid_index[:,1])
        y_top = np.max(valid_index[:,1])

    return [x_left, y_bottom, x_right, y_top]

class ARTODataset(BaseDataset):
    """A template dataset class for you to implement custom datasets."""
    def __init__(self, opt, is_for_train):
        """Initialize this dataset class.
        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        A few things can be done here.
        - save the options (have been done in BaseDataset)
        - get image paths and meta information of the dataset.
        - define the image transformation.
        """
        # save the option and dataset root
        BaseDataset.__init__(self, opt)
        ## content and mask
        self.path_content = []
        self.path_content_mask = []
        ## style
        self.path_style = []
        self.path_style_mask = []
        self.isTrain = is_for_train
        self.opt = opt
        self._load_images_paths()
        self.transform = get_transform(opt)
        

    def _load_images_paths(self,):
        if self.isTrain:
            print('loading training set')
            for style_folder in os.listdir(self.opt.info_dir):
                for painterly_object in os.listdir(os.path.join(self.opt.info_dir, style_folder)):
                    with open(os.path.join(self.opt.info_dir, style_folder, painterly_object), 'r', encoding='utf-8') as fr:
                        for line in fr.readlines():
                            name_parts = line.strip().split(',')
                            painterly_name = name_parts[0]
                            photographic_name = name_parts[-1]
                            self.path_style.append(os.path.join(self.opt.style_dir, painterly_name[:-6]+'.jpg'))
                            self.path_style_mask.append(os.path.join(self.opt.style_dir.replace('wikiart', 'wikiart_object_mask'), painterly_name))
                            self.path_content.append(os.path.join(self.opt.content_dir, 'object', photographic_name))
                            self.path_content_mask.append(os.path.join(self.opt.content_dir, 'mask', photographic_name.replace('.jpg', '.png')))
        else:
            print('loading testing set')

            for item in os.listdir(os.path.join(self.opt.info_dir, 'comp')):
                self.path_content.append(os.path.join(self.opt.info_dir, 'comp', item))
                self.path_content_mask.append(os.path.join(self.opt.info_dir, 'mask', item.replace('.jpg', '.png')))
                self.path_style.append(os.path.join(self.opt.info_dir, 'style', item))
                # we do not use style mask during test phase
                self.path_style_mask.append(os.path.join(self.opt.info_dir, 'mask', item.replace('.jpg', '.png')))


        print('foreground number',len(self.path_content))
        print('background number',len(self.path_style))
        

    def get_patch_mask(self, mask, number=4):
        """generate n*n patch to supervise discriminator"""
        mask = np.asarray(mask)
        mask = np.uint8(mask / 255.)
        mask_small = np.zeros([number, number],dtype=np.float32)
        split_size = self.opt.load_size // number
        for i in range(number):
            for j in range(number):
                mask_split = mask[i*split_size: (i+1)*split_size, j*split_size: (j+1)*split_size]
                mask_small[i, j] = (np.sum(mask_split) > 0) * 255
                #mask_small[i, j] = (np.sum(mask_split) / (split_size * split_size)) * 255
        mask_small = np.uint8(mask_small)
        return Image.fromarray(mask_small,mode='L')

    def __getitem__(self, index):
        if self.isTrain:
            content = cv2.cvtColor(cv2.imread(self.path_content[index]), cv2.COLOR_BGR2RGB)
            content_mask = cv2.imread(self.path_content_mask[index], 0)

            style = cv2.cvtColor(cv2.imread(self.path_style[index]), cv2.COLOR_BGR2RGB)
            style_mask = cv2.imread(self.path_style_mask[index], 0)
            style = cv2.resize(style, (self.opt.load_size, self.opt.load_size))
            style_mask = cv2.resize(style_mask, (self.opt.load_size, self.opt.load_size))
            style_mask_bbox = mask_bboxregion(style_mask)

            content = cv2.resize(content, (style_mask_bbox[3] - style_mask_bbox[1], style_mask_bbox[2] - style_mask_bbox[0]))
            content_mask = cv2.resize(content_mask, (style_mask_bbox[3] - style_mask_bbox[1], style_mask_bbox[2] - style_mask_bbox[0]))
            content_new, content_mask_new = np.zeros(style.shape), np.zeros(style_mask.shape)

            content_new[style_mask_bbox[0]: style_mask_bbox[2], style_mask_bbox[1]: style_mask_bbox[3], :] = content
            content_mask_new[style_mask_bbox[0]: style_mask_bbox[2], style_mask_bbox[1]: style_mask_bbox[3]] = content_mask
            content_patch_mask = self.get_patch_mask(content_mask_new)

            style, style_mask = Image.fromarray(style.astype(np.uint8)), Image.fromarray(style_mask.astype(np.uint8))
            content_new, content_mask_new = Image.fromarray(content_new.astype(np.uint8)), Image.fromarray(content_mask_new.astype(np.uint8))

            style = self.transform(style)
            content_new = self.transform(content_new)

            style_mask = tf.to_tensor(style_mask)
            content_mask_new = tf.to_tensor(content_mask_new)
            content_patch_mask = tf.to_tensor(content_patch_mask)

            style_mask = style_mask*2 -1
            content_mask_new = content_mask_new*2 -1
            content_patch_mask = content_patch_mask*2 -1

            comp = self._compose(content_new, content_mask_new, style)

            return {'content': content_new, 'comp': comp, 'comp_mask': content_mask_new, 'comp_patch_mask': content_patch_mask, \
                    'style': style, 'style_mask': style_mask, \
                    'content_path':self.path_content_mask[index], 'style_path':self.path_style_mask[index]}
        else:
            content = cv2.cvtColor(cv2.imread(self.path_content[index]), cv2.COLOR_BGR2RGB)
            content_mask = cv2.imread(self.path_content_mask[index], 0)
            content = cv2.resize(content, (self.opt.load_size, self.opt.load_size))
            content_mask = cv2.resize(content_mask, (self.opt.load_size, self.opt.load_size))
            content_patch_mask = self.get_patch_mask(content_mask)

            style = cv2.cvtColor(cv2.imread(self.path_style[index]), cv2.COLOR_BGR2RGB)
            style_mask = cv2.imread(self.path_style_mask[index], 0)
            style = cv2.resize(style, (self.opt.load_size, self.opt.load_size))
            style_mask = cv2.resize(style_mask, (self.opt.load_size, self.opt.load_size))

            style, style_mask = Image.fromarray(style.astype(np.uint8)), Image.fromarray(style_mask.astype(np.uint8))
            content, content_mask = Image.fromarray(content.astype(np.uint8)), Image.fromarray(content_mask.astype(np.uint8))

            style = self.transform(style)
            content = self.transform(content)

            style_mask = tf.to_tensor(style_mask)
            content_mask = tf.to_tensor(content_mask)
            content_patch_mask = tf.to_tensor(content_patch_mask)

            style_mask = style_mask*2 -1
            content_mask = content_mask*2 -1
            content_patch_mask = content_patch_mask*2 -1

            comp = self._compose(content, content_mask, style)

            return {'content': content, 'comp': comp, 'comp_mask': content_mask, 'comp_patch_mask': content_patch_mask, \
                    'style': style, 'style_mask': style_mask, \
                    'content_path':self.path_content_mask[index], 'style_path':self.path_style_mask[index]}

    def __len__(self):
        return len(self.path_style)

    def _compose(self, foreground_img, foreground_mask, background_img):
        foreground_img = foreground_img/2 + 0.5
        background_img = background_img/2 + 0.5
        foreground_mask = foreground_mask/2 + 0.5
        comp = foreground_img * foreground_mask + background_img * (1 - foreground_mask)
        comp = comp*2-1
        return comp

    def _compose_cover(self, foreground_img, foreground_mask, foreground_bbox, background_img, background_bbox):
        torch_resize = Resize([background_bbox[2]-background_bbox[0],background_bbox[3]-background_bbox[1]])
        
        # x_left, y_top, x_right, y_bottom
        foreground_img = foreground_img/2 + 0.5
        background_img = background_img/2 + 0.5
        foreground_mask = foreground_mask/2 + 0.5
        #print(foreground_bbox, background_bbox)

        # crop then resize foreground mask
        fg = foreground_img[:,foreground_bbox[0]:foreground_bbox[2], foreground_bbox[1]:foreground_bbox[3]]
        fg_mask_region = foreground_mask[:,foreground_bbox[0]:foreground_bbox[2], foreground_bbox[1]:foreground_bbox[3]]
        #print(background_bbox[2]-background_bbox[0], background_bbox[3]-background_bbox[1], bg_mask_region.shape)
        fg_mask_region = torch_resize(fg_mask_region)
        fg = torch_resize(fg)
        fg_mask_new = torch.zeros(foreground_mask.size())
        fg_mask_new[:,background_bbox[0]:background_bbox[2], background_bbox[1]:background_bbox[3]] = fg_mask_region
        # crop then resize foreground object
        fg_new = torch.zeros(foreground_img.size())
        fg_new[:,background_bbox[0]:background_bbox[2], background_bbox[1]:background_bbox[3]] = fg

        comp = fg_new * fg_mask_new + background_img * (1 - fg_mask_new)
        comp = comp*2-1
        fg_mask_new = fg_mask_new*2-1
        fg_new = fg_new*2-1
        return comp, fg_mask_new, fg_new