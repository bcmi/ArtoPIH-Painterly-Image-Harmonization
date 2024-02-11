import os
from os.path import realpath
from options.test_options import TestOptions
import torch
import numpy as np
from util import util
from util.visualizer import Visualizer
from PIL import Image, ImageFile
from data import CustomDataset
from models import create_model
from tqdm import tqdm
import time
import shutil


opt = TestOptions().parse()   # get training 
opt.isTrain = False
visualizer = Visualizer(opt,'test')
test_dataset = CustomDataset(opt, is_for_train=False)
test_dataset_size = len(test_dataset)
print('The number of testing images = %d' % test_dataset_size)
test_dataloader = test_dataset.load_data()

model = create_model(opt)      # create a model given opt.model and other options
model.setup(opt)               # regular setup: load and print networks; create schedulers
model.netG.eval()
total_iters = 0


for i, data in enumerate(tqdm(test_dataloader)):  # inner loop within one epoch

    img_path = data['style_path'][0]
    save_dir = '../output'
    img_name = img_path.split('/')[-1]

    model.set_input(data)         # unpack data from dataset
    model.test()   # calculate loss functions, get gradients, update network weights
    visual_dict = model.get_current_visuals()
    
    print('saving iteration {}'.format(i))
    # visualizer.display_current_results(visual_dict, total_iters)
    visualizer.save_images(visual_dict, save_dir, img_name)
