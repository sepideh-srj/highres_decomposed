import torch
from .base_model import BaseModel
from . import networks


class HighResModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(norm='batch', netG='unet_256', dataset_mode='aligned')
        parser.add_argument('--ratio', type=float, default=1.0, help='ratio image will be used in loss functions')
        parser.add_argument('--lambda_feat', type=float, default=10)
        parser.add_argument('--no_vgg_loss', action='store_true')
        parser.add_argument('--lambda_gradient', type=float, default=10)
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'G_L1', 'D_real', 'D_fake']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['ratio_low', 'ratio_real', 'fake_ratio', 'input_real', 'real_output', 'fake_output']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G', 'D']
            if not self.opt.no_vgg_loss:
                self.loss_names += ['VGG']
            if self.opt.lambda_gradient:
                self.loss_names += ['G_gradient']
        else:  # during test time, only load G
            self.model_names = ['G']
        # define networks (both generator and discriminator)
        if self.isTrain:
            # define a discriminator; conditional GANs need to take both input and output images; Therefore,
            # #channels for D is input_nc + output_nc
            self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            if self.opt.gan_mode == 'wgangp':
                self.loss_names += ['gradient_penalty']
        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            if self.opt.gan_mode == 'vannilla':
                self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            else:
                self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr2, betas=(opt.beta1, 0.999))
            if not opt.no_vgg_loss:
                self.criterionVGG = networks.VGGLoss(self.gpu_ids)
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):

        AtoB = self.opt.direction == 'AtoB'
        self.ratio_low = input['A_ratio_low' if AtoB else 'B_ratio_low'].to(self.device)
        self.ratio_real = input['A_ratio_high' if AtoB else 'B_ratio_high'].to(self.device)
        self.input_real = input['ambient_highRes' if AtoB else 'flashPhoto_highRes'].to(self.device)
        self.input_low = input['flashPhoto_lowRes' if AtoB else 'ambient_lowRes'].to(self.device)
        self.gr_high = input['flashPhoto_highRes' if AtoB else 'ambient_highRes'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""''
        self.input = torch.cat((self.input_real, self.ratio_low), 1)
        self.fake_ratio = self.netG(self.input)  # G(A)
        # self.real_output = (2 * (self.input_real + 1)) / (3 * self.fake_ratio + 1) - 1
        if not self.opt.ratio:
            self.fake_output = (
                                       3 * self.input_real * self.fake_ratio + 9 * self.fake_ratio + 5 * self.input_real + 3) / 4

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        if self.opt.ratio:
            fake_AB = torch.cat((self.input, self.fake_ratio), 1)
        else:
            fake_AB = torch.cat((self.input, self.fake_output),
                                1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        # Real
        if self.opt.ratio:
            real_AB = torch.cat((self.input, self.ratio_real), 1)
        else:
            real_AB = torch.cat((self.input, self.gr_high), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        if (self.opt.gan_mode == 'wgangp'):
            self.loss_gradient_penalty, gradients = networks.cal_gradient_penalty(
                self.netD, real_AB, fake_AB, self.device, lambda_gp=20.0
            )
            self.loss_gradient_penalty.backward(retain_graph=True)

        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        if self.opt.ratio:
            fake_AB = torch.cat((self.input, self.fake_ratio), 1)
        else:
            fake_AB = torch.cat((self.input, self.fake_output), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        # Second, G(A) = B
        if self.opt.ratio:
            self.loss_G_L1 = self.criterionL1(self.fake_ratio, self.ratio_real) * self.opt.lambda_L1
        else:
            self.loss_G_L1 = self.criterionL1(self.fake_output, self.gr_high) * self.opt.lambda_L1

        self.loss_VGG = 0
        if not self.opt.no_vgg_loss:
            if self.opt.ratio:
                self.fake_output = (
                                               3 * self.input_real * self.fake_ratio + 9 * self.fake_ratio + 5 * self.input_real + 3) / 4
            self.loss_VGG = self.criterionVGG(self.fake_output, self.gr_high) * self.opt.lambda_feat

        self.loss_G_gradient = 0
        if self.opt.lambda_gradient:
            if self.opt.ratio:
                self.fake_output = (
                                           3 * self.input_real * self.fake_ratio + 9 * self.fake_ratio + 5 * self.input_real + 3) / 4
            gt = self.gr_high
            pred = self.fake_output
            reduction = reduction_batch_based
            self.loss_G_gradient = 0.0
            for ds in range(4):
                scale = pow(2, ds)
                pred_scale = pred[:, :, ::scale, ::scale]
                gt_scale = gt[:, :, ::scale, ::scale]

                self.loss_G_gradient += self.gradient_loss(pred_scale,
                                                           gt_scale, reduction) * self.opt.lambda_gradient
        # combine loss and calculate gradients
        self.loss_G = self.loss_G_GAN + self.loss_G_L1 + self.loss_VGG + self.loss_G_gradient
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()  # compute fake images: G(A)
        # update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()  # set D's gradients to zero
        self.backward_D()  # calculate gradients for D
        self.optimizer_D.step()  # update D's weights
        # update G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()  # set G's gradients to zero
        self.backward_G()  # calculate graidents for G
        self.optimizer_G.step()  # udpate G's weights

    def gradient_loss(self, prediction, target, reduction):
        M = torch.sum(target, (2, 3))

        diff = prediction - target
        grad_x = torch.abs(diff[:, :, :, 1:] - diff[:, :, :, :-1])
        grad_y = torch.abs(diff[:, :, 1:, :] - diff[:, :, :-1, :])
        image_loss = torch.sum(grad_x, (2, 3)) + torch.sum(grad_y, (2, 3))

        return reduction(image_loss, M)


def reduction_batch_based(image_loss, M):
    # average of all valid pixels of the batch

    # avoid division by 0 (if sum(M) = sum(sum(mask)) = 0: sum(image_loss) = 0)
    divisor = torch.sum(M)

    if divisor == 0:
        return 0
    else:
        return torch.abs(torch.sum(image_loss) / divisor)
