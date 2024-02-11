import time
from options.train_options import TrainOptions
from data import CustomDataset
from models import create_model
# from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter

import os
from util import util
import numpy as np
import torch
from skimage.metrics import mean_squared_error
from skimage.metrics import peak_signal_noise_ratio
from tqdm import tqdm

from util.visualizer import Visualizer
from PIL import Image, ImageFile


Image.MAX_IMAGE_PIXELS = None  # Disable DecompressionBombError
# Disable OSError: image file is truncated
ImageFile.LOAD_TRUNCATED_IMAGES = True

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    

if __name__ == '__main__':
    # setup_seed(6)

    opt = TrainOptions().parse()   # get training 
    visualizer = Visualizer(opt,'train')
    visualizer_test = Visualizer(opt,'test')


    train_dataset = CustomDataset(opt, is_for_train=True)
    # test_dataset = CustomDataset(opt, is_for_train=False)

    train_dataset_size = len(train_dataset)    # get the number of images in the dataset.
    # test_dataset_size = len(test_dataset)
    print('The number of training images = %d' % train_dataset_size)
    # print('The number of testing images = %d' % test_dataset_size)
    
    train_dataloader = train_dataset.load_data()
    # test_dataloader = test_dataset.load_data()
    print('The total batches of training images = %d' % len(train_dataset.dataloader))

    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    total_iters = 0                # the total number of training iterations
    writer = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.name))

    for epoch in tqdm(range(int(opt.load_iter)+1, opt.niter + opt.niter_decay + 1)):
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch

        for i, data in enumerate(tqdm(train_dataloader)):  # inner loop within one epoch
            if i > len(train_dataset.dataloader) -2:
                continue
            # print('epoch {} iter {}'.format(epoch, i))
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            total_iters += 1
            epoch_iter += 1
            model.set_input(data)         # unpack data from dataset
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

            if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                writer.add_scalar('./loss/loss_c', losses['c'], total_iters + 1)
                writer.add_scalar('./loss/loss_s', losses['s'], total_iters + 1)
                writer.add_scalar('./loss/loss_rec',losses['rec'], total_iters + 1)
                writer.add_scalar('./loss/loss_class', losses['class'], total_iters + 1)
                writer.add_scalar('./loss/loss_tv', losses['tv'], total_iters + 1)
            
            if total_iters % opt.display_freq == 0:
                visual_dict = model.get_current_visuals()
                visualizer.display_current_results(visual_dict, epoch)
                # visualizer.display_current_results(visual_dict, total_iters)

                
            if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix, os.path.join(opt.checkpoints_dir, opt.name))

            iter_data_time = time.time()


        torch.cuda.empty_cache()
        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest', os.path.join(opt.checkpoints_dir, opt.name))
            model.save_networks('%d' % epoch, os.path.join(opt.checkpoints_dir, opt.name))

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        model.update_learning_rate()                     # update learning rates at the end of every epoch.
        for scheduler in model.schedulers:
            print('Current learning rate: {}'.format(scheduler.get_lr()))

    writer.close()
