import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import functools
from torch.optim import lr_scheduler

from util.losses import TVLoss
import time

  
def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.contiguous().view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.contiguous().view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std

def get_foreground_mean_std(features, mask, eps=1e-5):
    region = features * mask 
    sum = torch.sum(region, dim=[2, 3])     # (B, C)
    num = torch.sum(mask, dim=[2, 3])       # (B, C)
    mu = sum / (num + eps)
    mean = mu[:, :, None, None]
    var = torch.sum((region + (1 - mask)*mean - mean) ** 2, dim=[2, 3]) / (num + eps)
    var = var[:, :, None, None]
    std = torch.sqrt(var+eps)
    return mean, std


decoder = nn.Sequential(
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 256, (3, 3)),
    nn.ReLU(), # relu1-1
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),   # 4
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 128, (3, 3)),
    nn.ReLU(), # relu2-1
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),  # 17
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 64, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),  # 24
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),
    # nn.Conv2d(64, 1, (3, 3),padding=0,stride=1), ##matting layer
    nn.Conv2d(65, 1, (1, 1),padding=0,stride=1), ##matting layer  # 27
    nn.ReflectionPad2d((1, 1, 1, 1)),
    # nn.ReflectionPad2d((1, 1, 1, 1)), ##matting layer
    nn.Conv2d(64, 3, (3, 3)),
)

decoder_cat = nn.Sequential(
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 256, (3, 3)),
    nn.ReLU(), # relu1-1
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),  # 4
    nn.Conv2d(512, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 128, (3, 3)),
    nn.ReLU(), # relu2-1
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),  # 17
    nn.Conv2d(256, 128, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 64, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),  # 24
    nn.Conv2d(128, 64, (3, 3)),
    nn.ReLU(),
    nn.Conv2d(65, 1, (1, 1),padding=0,stride=1), ##matting layer
    nn.ReflectionPad2d((1, 1, 1, 1)), # 29
    nn.Conv2d(64, 3, (3, 3)),
)


vgg = nn.Sequential(
    nn.Conv2d(3, 3, (1, 1)),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(3, 64, (3, 3)),
    nn.ReLU(),  # relu1-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),  # relu1-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 128, (3, 3)),
    nn.ReLU(),  # relu2-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),  # relu2-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 256, (3, 3)),
    nn.ReLU(),  # relu3-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 512, (3, 3)),
    nn.ReLU(),  # relu4-1, this is the last layer used
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU()  # relu5-4
)


class Identity(nn.Module):
    def forward(self, x):
        return x

def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    norm_type = norm_type.lower()
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        norm_layer = lambda x: Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer



def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.niter> epochs
    and linearly decay the rate to zero over the next <opt.niter_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler

def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>

def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'nsgan':
            self.loss = nn.BCELoss()
        elif gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
            self.relu = nn.ReLU()
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['nsgan', 'lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean() # self.relu(1-prediction.mean())
            else:
                loss = prediction.mean() # self.relu(1+prediction.mean())
        return loss

def cal_gradient_penalty(netD, real_data, fake_data, device, type='mixed', constant=1.0, lambda_gp=10.0, mask=None):
    """Calculate the gradient penalty loss, used in WGAN-GP paper https://arxiv.org/abs/1704.00028

    Arguments:
        netD (network)              -- discriminator network
        real_data (tensor array)    -- real images
        fake_data (tensor array)    -- generated images from the generator
        device (str)                -- GPU / CPU: from torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        type (str)                  -- if we mix real and fake data or not [real | fake | mixed].
        constant (float)            -- the constant used in formula ( | |gradient||_2 - constant)^2
        lambda_gp (float)           -- weight for this loss

    Returns the gradient penalty loss
    """
    if lambda_gp > 0.0:
        if type == 'real':   # either use real images, fake images, or a linear interpolation of two.
            interpolatesv = real_data
        elif type == 'fake':
            interpolatesv = fake_data
        elif type == 'mixed':
            alpha = torch.rand(real_data.shape[0], 1, device=device)
            alpha = alpha.unsqueeze(2).unsqueeze(3)
            alpha = alpha.expand_as(real_data)
            interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
        else:
            raise NotImplementedError('{} not implemented'.format(type))
        interpolatesv.requires_grad_(True)
        disc_interpolates = netD(interpolatesv, mask, gp=True)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv,
                                        grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                        create_graph=True, retain_graph=True, only_inputs=True,
                                        allow_unused=True)
        gradients = gradients[0].view(real_data.size(0), -1)  # flat the data
        gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp        # added eps
        return gradient_penalty, gradients
    else:
        return 0.0, None



class LinearBlock(nn.Module):
    def __init__(self, in_dim, out_dim, norm='none', activation='relu'):
        super(LinearBlock, self).__init__()
        use_bias = True
        self.fc = nn.Linear(in_dim, out_dim, bias=use_bias)

        # initialize normalization
        norm_dim = out_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm1d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm1d(norm_dim)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

    def forward(self, x):
        out = self.fc(x)
        if self.norm:
            out = self.norm(out)
        if self.activation:
            out = self.activation(out)
        return out


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, dim, n_blk, norm, activ):
        super(MLP, self).__init__()
        self.model = []
        self.model += [LinearBlock(in_dim, dim, norm=norm, activation=activ)]
        for i in range(n_blk - 2):
            self.model += [LinearBlock(dim, dim, norm=norm, activation=activ)]
        self.model += [LinearBlock(dim, out_dim,
                                   norm='none', activation='none')]
        self.model = nn.Sequential(*self.model)

    def forward(self, style_vector, content_vector):
        n,c,h,w = style_vector.size()
        x = torch.cat([style_vector, content_vector],1)
        y = self.model(x.view(n,-1))
        return y.view(n,c,1,1)


class ArtoNet(nn.Module):
    def __init__(self, encoder, decoder, is_matting=True):
        super(ArtoNet, self).__init__()
        # print('obfinetune')
        enc_layers = list(encoder.children())
        self.enc_1 = nn.Sequential(*enc_layers[:4])  # input -> relu1_1
        self.enc_2 = nn.Sequential(*enc_layers[4:11])  # relu1_1 -> relu2_1
        self.enc_3 = nn.Sequential(*enc_layers[11:18])  # relu2_1 -> relu3_1
        self.enc_4 = nn.Sequential(*enc_layers[18:31])  # relu3_1 -> relu4_1
        
        # define the losses
        self.mse_loss = nn.MSELoss()
        self.entropy_loss = nn.CrossEntropyLoss()
        self.tv_loss = TVLoss(1)

        # define the decoder
        self.decoder = decoder
        dec_layers = list(decoder.children())
        self.dec_1 =  nn.Sequential(*dec_layers[:4]) 
        self.dec_2 =  nn.Sequential(*dec_layers[4:17]) 
        self.dec_3 =  nn.Sequential(*dec_layers[17:24]) 
        self.dec_4 =  nn.Sequential(*dec_layers[24:27])   
        self.conv_attention = nn.Sequential(*dec_layers[27:28]) 
        self.dec_4_2 =  nn.Sequential(*dec_layers[28:])
        self.is_matting = is_matting

        ## define domain-invariant content module
        self.content_extractor = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True),
            )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)


        # define MLP in ObAdain
        self.MLP_4 = MLP(64*2+256, 64*2, 128*2, 3, norm='none', activ='relu')
        self.MLP_3 = MLP(128*2+256, 128*2, 256*2, 3, norm='none', activ='relu')
        self.MLP_2 = MLP(256*2+256, 256*2, 512*2, 3, norm='none', activ='relu')
        self.MLP_1 = MLP(512*2+256, 512*2, 1024*2, 3, norm='none', activ='relu')


        # fix the encoder
        for name in ['enc_1', 'enc_2', 'enc_3', 'enc_4']:
            for param in getattr(self, name).parameters():
                param.requires_grad = False

    # extract relu1_1, relu2_1, relu3_1, relu4_1 from input image
    def encode_with_intermediate(self, input):
        results = [input]
        for i in range(4):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]
    
    def get_content_feature(self, feature, mask):
        feat = self.content_extractor(feature)
        width = height = feat.size(-1)
        downsample_mask = self.downsample(mask, width, height)
        content_v = self.avg_pool(feat*downsample_mask)
        return content_v


    def decode_train(self, comp, style, comp_mask, style_mask, comp_feats, style_feats):
        ## extract domain-invariant content feature
        comp_content_feat = self.get_content_feature(comp_feats[-1], comp_mask)
        style_content_feat = self.get_content_feature(style_feats[-1], style_mask)
        ## calculate loss for content_extractor
        loss_class = self.mse_loss(comp_content_feat, style_content_feat)

        dim = comp_feats[-1].size(1)
        width = height = comp_feats[-1].size(-1)
        ## calculate style vector for style image
        downsample_style_mask = self.downsample(style_mask, width, height)
        style_fg_mu, style_fg_sigma = get_foreground_mean_std(style_feats[-1], downsample_style_mask)
        style_bg_mu, style_bg_sigma = get_foreground_mean_std(style_feats[-1], 1-downsample_style_mask)
        style_fg_v = torch.cat([style_fg_mu, style_fg_sigma], 1)
        style_bg_v = torch.cat([style_bg_mu, style_bg_sigma], 1)
        ## calculate style vector for composite image
        downsample_comp_mask = self.downsample(comp_mask, width, height)
        downsample_allone_mask = torch.ones(downsample_comp_mask.size()).to(downsample_comp_mask.device)
        comp_fg_mu, comp_fg_sigma = get_foreground_mean_std(comp_feats[-1], downsample_comp_mask)

        ## ObAdain
        harm_fg_v = self.MLP_1(style_bg_v, comp_content_feat)
        harm_fg_mu, harm_fg_sigma = harm_fg_v[:,:dim,:,:], harm_fg_v[:,dim:,:,:]
        # channel-wise feature alignment
        norm_feat = (comp_feats[-1] - comp_fg_mu) / comp_fg_sigma
        harm_feat = (norm_feat * harm_fg_sigma + harm_fg_mu) * downsample_comp_mask + (comp_feats[-1] * (1 - downsample_comp_mask))

        dec_feat = self.dec_1(harm_feat)
        ## calculate loss for ObAdain
        style_fg_v_rec = self.MLP_1(style_bg_v, style_content_feat)
        loss_rec = self.mse_loss(style_fg_v_rec, style_fg_v) + self.mse_loss(harm_fg_v, style_fg_v)

        ## ObAdain for each layer
        for i in range(1, 4):
            decoder = getattr(self, 'dec_{:d}'.format(i + 1))
            mlp = getattr(self, 'MLP_{:d}'.format(i + 1))
            dim = comp_feats[-(i+1)].size(1)
            width = height = comp_feats[-(i+1)].size(-1)
            ## calculate style vector for style image
            downsample_style_mask = self.downsample(style_mask, width, height)
            style_fg_mu, style_fg_sigma = get_foreground_mean_std(style_feats[-(i+1)], downsample_style_mask)
            style_bg_mu, style_bg_sigma = get_foreground_mean_std(style_feats[-(i+1)], 1-downsample_style_mask)
            style_fg_v = torch.cat([style_fg_mu, style_fg_sigma], 1)
            style_bg_v = torch.cat([style_bg_mu, style_bg_sigma], 1)
            ## calculate style vector for composite image
            downsample_comp_mask = self.downsample(comp_mask, width, height)
            downsample_allone_mask = torch.ones(downsample_comp_mask.size()).to(comp.device)
            comp_fg_mu, comp_fg_sigma = get_foreground_mean_std(comp_feats[-(i+1)], downsample_comp_mask)

            ## ObAdain
            harm_fg_v = mlp(style_bg_v, comp_content_feat)
            harm_fg_mu, harm_fg_sigma = harm_fg_v[:,:dim,:,:], harm_fg_v[:,dim:,:,:]
            # channel-wise feature alignment
            norm_feat = (comp_feats[-(i+1)] - comp_fg_mu) / comp_fg_sigma
            harm_feat = (norm_feat * harm_fg_sigma + harm_fg_mu) * downsample_comp_mask + (comp_feats[-(i+1)] * (1 - downsample_comp_mask))

            dec_feat = decoder(torch.cat([dec_feat, harm_feat], dim=1))
            ## calculate loss for ObAdain
            style_fg_v_rec = mlp(style_bg_v, style_content_feat)
            loss_rec += self.mse_loss(style_fg_v_rec, style_fg_v) + self.mse_loss(harm_fg_v, style_fg_v)

        if self.is_matting:
            width = height = dec_feat.size(-1)
            downsample_comp_mask = self.downsample(comp_mask, width, height)
            attention_mask = torch.sigmoid(self.conv_attention(torch.cat((dec_feat,downsample_comp_mask),dim=1)))
            coarse_output = self.dec_4_2(dec_feat)
            output = attention_mask * coarse_output + (1.0 - attention_mask) * comp
            loss_mask = self.mse_loss(attention_mask, comp_mask)
            return output, coarse_output, attention_mask, loss_rec, loss_class, loss_mask, comp_content_feat, style_content_feat
        else:
            coarse_output = self.dec_4_2(dec_feat)
            output = comp * (1 - comp_mask) + coarse_output * comp_mask
            return output, coarse_output, loss_rec, loss_class, comp_content_feat, style_content_feat

    def decode_test(self, comp, comp_mask, comp_feats, style, style_mask, style_feats):
        ## extract domain-invariant content feature
        comp_content_feat = self.get_content_feature(comp_feats[-1], comp_mask)
        style_content_feat = self.get_content_feature(style_feats[-1], style_mask)
        ## calculate style vector for composite image
        dim = comp_feats[-1].size(1)
        width = height = comp_feats[-1].size(-1)
        downsample_comp_mask = self.downsample(comp_mask, width, height)
        downsample_allone_mask = torch.ones(downsample_comp_mask.size()).to(comp.device)
        comp_fg_mu, comp_fg_sigma = get_foreground_mean_std(comp_feats[-1], downsample_comp_mask)
        ## calculate style vector for style image
        downsample_style_mask = self.downsample(style_mask, width, height)
        style_fg_mu, style_fg_sigma = get_foreground_mean_std(style_feats[-1], downsample_style_mask)
        #style_bg_mu, style_bg_sigma = get_foreground_mean_std(style_feats[-1], downsample_allone_mask)
        style_bg_mu, style_bg_sigma = get_foreground_mean_std(style_feats[-1], 1-downsample_style_mask)
        style_fg_v = torch.cat([style_fg_mu, style_fg_sigma], 1)
        style_bg_v = torch.cat([style_bg_mu, style_bg_sigma], 1)

        ## ObAdain
        harm_fg_v = self.MLP_1(style_bg_v, comp_content_feat)
        harm_fg_mu, harm_fg_sigma = harm_fg_v[:,:dim,:,:], harm_fg_v[:,dim:,:,:]
        # channel-wise feature alignment
        norm_feat = (comp_feats[-1] - comp_fg_mu) / comp_fg_sigma
        harm_feat = (norm_feat * harm_fg_sigma + harm_fg_mu) * downsample_comp_mask + (comp_feats[-1] * (1 - downsample_comp_mask))
        
        dec_feat = self.dec_1(harm_feat)

        ## ObAdain for each layer
        for i in range(1, 4):
            decoder = getattr(self, 'dec_{:d}'.format(i + 1))
            mlp = getattr(self, 'MLP_{:d}'.format(i + 1))
            ## calculate style vector for composite image
            dim = comp_feats[-(i+1)].size(1)
            width = height = comp_feats[-(i+1)].size(-1)
            downsample_comp_mask = self.downsample(comp_mask, width, height)
            downsample_allone_mask = torch.ones(downsample_comp_mask.size()).to(comp.device)
            comp_fg_mu, comp_fg_sigma = get_foreground_mean_std(comp_feats[-(i+1)], downsample_comp_mask)
            ## calculate style vector for style image
            downsample_style_mask = self.downsample(style_mask, width, height)
            style_fg_mu, style_fg_sigma = get_foreground_mean_std(style_feats[-(i+1)], downsample_style_mask)
            #style_bg_mu, style_bg_sigma = get_foreground_mean_std(style_feats[-(i+1)], downsample_allone_mask)
            style_bg_mu, style_bg_sigma = get_foreground_mean_std(style_feats[-(i+1)], 1-downsample_style_mask)
            style_fg_v = torch.cat([style_fg_mu, style_fg_sigma], 1)
            style_bg_v = torch.cat([style_bg_mu, style_bg_sigma], 1)

            ## ObAdain
            harm_fg_v = mlp(style_bg_v, comp_content_feat)
            harm_fg_mu, harm_fg_sigma = harm_fg_v[:,:dim,:,:], harm_fg_v[:,dim:,:,:]
            # channel-wise feature alignment
            norm_feat = (comp_feats[-(i+1)] - comp_fg_mu) / comp_fg_sigma
            harm_feat = (norm_feat * harm_fg_sigma + harm_fg_mu) * downsample_comp_mask + (comp_feats[-(i+1)] * (1 - downsample_comp_mask))

            dec_feat = decoder(torch.cat([dec_feat, harm_feat], dim=1))

        if self.is_matting:
            width = height = dec_feat.size(-1)
            downsample_comp_mask = self.downsample(comp_mask, width, height)
            attention_mask = torch.sigmoid(self.conv_attention(torch.cat((dec_feat, downsample_comp_mask),dim=1)))
            coarse_output = self.dec_4_2(dec_feat)
            output = attention_mask * coarse_output + (1.0 - attention_mask) * comp
            return output, coarse_output, attention_mask, comp_content_feat, style_content_feat
        else:
            coarse_output = self.dec_4_2(dec_feat)
            output = comp * (1 - comp_mask) + coarse_output * comp_mask
            return output, coarse_output, comp_content_feat, style_content_feat

    # extract relu4_1 from input image
    def encode(self, input):
        for i in range(4):
            input = getattr(self, 'enc_{:d}'.format(i + 1))(input)
        return input

    def calc_content_loss(self, gen, comb):
        loss =  self.mse_loss(gen, comb) 
        # assert (target.requires_grad is False)
        return loss

    def downsample(self, image_tensor, width, height):
        image_upsample_tensor = torch.nn.functional.interpolate(image_tensor, size=[width, height])
        image_upsample_tensor = image_upsample_tensor.clamp(0, 1)
        return image_upsample_tensor


    def calc_style_loss_fg_fg(self, comps, styles, comp_mask, style_mask):
        loss = torch.zeros(1).to(comp_mask.device)
        for i in range(0, 4):
            width = height = comps[i].size(-1)
            downsample_comp_mask = self.downsample(comp_mask, width, height)
            downsample_mask_style = self.downsample(style_mask, width, height)

            mu_cs, sigma_cs = get_foreground_mean_std(comps[i], downsample_comp_mask)
            mu_target,sigma_target = get_foreground_mean_std(styles[i], downsample_mask_style)
            loss_i = self.mse_loss(mu_cs, mu_target) + self.mse_loss(sigma_cs, sigma_target)
            loss += loss_i
        return loss

    def __call__(self, comp, style, comp_mask, style_mask, isTrain=False):
        if isTrain:
            return self.forward_train(comp, style, comp_mask, style_mask)
        else:
            return self.forward_test(comp, comp_mask, style, style_mask)

    def forward_train(self, comp, style, comp_mask, style_mask):
        style_feats = self.encode_with_intermediate(style)
        comp_feats = self.encode_with_intermediate(comp)

        if self.is_matting:
            final_output, coarse_output, attention_mask, loss_rec, loss_class, loss_mask, comp_content_feat, style_content_feat = \
                self.decode_train(comp, style, comp_mask, style_mask, comp_feats, style_feats)
        else:
            final_output, coarse_output, loss_rec, loss_class, comp_content_feat, style_content_feat  = \
                self.decode_train(comp, style, comp_mask, style_mask, comp_feats, style_feats)
            attention_mask = comp_mask
            loss_mask = torch.zeros(1).to(comp.device)
        coarse_feats = self.encode_with_intermediate(coarse_output)
        fine_feats = self.encode_with_intermediate(final_output)
        # calculate content loss
        loss_c = self.calc_content_loss(coarse_feats[-1], comp_feats[-1])
        loss_c += self.calc_content_loss(fine_feats[-1], comp_feats[-1])
        # calculate style loss
        loss_s = self.calc_style_loss_fg_fg(coarse_feats, style_feats, comp_mask, style_mask)
        loss_s += self.calc_style_loss_fg_fg(fine_feats, style_feats, comp_mask, style_mask)
        # calculate smooth loss
        loss_tv = self.tv_loss(final_output)

        return coarse_output, final_output, attention_mask*2-1, loss_c, loss_s, loss_rec, loss_class, loss_tv, loss_mask, comp_content_feat, style_content_feat
    
    def forward_test(self, comp, comp_mask, style, style_mask):
        comp_feats = self.encode_with_intermediate(comp)
        style_feats = self.encode_with_intermediate(style)
        if self.is_matting:
            final_output, coarse_output, attention_mask, comp_content_feat, style_content_feat = self.decode_test(comp, comp_mask, comp_feats, style, style_mask, style_feats)
        else:
            final_output, coarse_output, comp_content_feat, style_content_feat = self.decode_test(comp, comp_mask, comp_feats, style, style_mask, style_feats)
            attention_mask = comp_mask

        return coarse_output, final_output, attention_mask*2-1, comp_content_feat, style_content_feat
