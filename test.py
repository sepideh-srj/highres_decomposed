"""General-purpose test script for image-to-image translation.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from '--checkpoints_dir' and save the results to '--results_dir'.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for '--num_test' images and save results to an HTML file.

Example (You need to train models first or download pre-trained models from our website):
    Test a CycleGAN model (both sides):
        python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan

    Test a CycleGAN model (one side only):
        python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout

    The option '--model test' is used for generating CycleGAN results only for one side.
    This option will automatically set '--dataset_mode single', which only loads the images from one set.
    On the contrary, using '--model cycle_gan' requires loading and generating results in both directions,
    which is sometimes unnecessary. The results will be saved at ./results/.
    Use '--results_dir <directory_path_to_save_result>' to specify the results directory.

    Test a pix2pix model:
        python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/test_options.py for more test options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
import torch
import numpy as np
import cv2
def gama_corect(rgb):
    srgb = np.zeros_like(rgb)
    mask1 = (rgb > 0) * (rgb < 0.0031308)
    mask2 = (1 - mask1).astype(bool)
    srgb[mask1] = 12.92 * rgb[mask1]
    srgb[mask2] = 1.055 * np.power(rgb[mask2], 0.41666) - 0.055
    srgb[srgb < 0] = 0
    srgb[srgb > 1] = 1
    return srgb

if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 0
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    # create a website
    web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.epoch))  # define the website directory
    if opt.load_iter > 0:  # load_iter is 0 by default
        web_dir = '{:s}_iter{:d}'.format(web_dir, opt.load_iter)
    print('creating web directory', web_dir)
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
    # test with eval mode. This only affects layers like batchnorm and dropout.
    # For [pix2pix]: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
    # For [CycleGAN]: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.
    if opt.eval:
        model.eval()
    for i, data in enumerate(dataset):
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break
        model.set_input(data)  # unpack data from data loader
        # model.test()           # run inference
        # visuals = model.get_current_visuals()  # get image results
        # img_path = model.get_image_paths()     # get image paths

        # if i % 5 == 0:  # save images to an HTML file
        #     print('processing (%04d)-th image... %s' % (i, img_path))
        # save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)
        # webpage.save()  # save the HTML
        with torch.no_grad():
            model.forward()  # run inference
        img_path = data['A_paths'][0]
        img_path = img_path.replace(opt.dataroot, '')
        seprator = '_'
        img_path = seprator.join(img_path.split('/'))
        # img_path = img_path.replace('.png', '')

        print(img_path)
        # ratio_real = data['A_ratio_high'].cpu().squeeze().numpy().astype('float32')

        ratio_low = model.ratio_low.cpu().squeeze().numpy().astype('float32')
        ratio_real = model.ratio_real.cpu().squeeze().numpy().astype('float32')
        input_real = model.input_real.cpu().squeeze().numpy().astype('float32')
        output_real = model.gr_high.cpu().squeeze().numpy().astype('float32')

        if opt.ratio == 1:
            fake_ratio = model.fake_ratio.cpu().squeeze().numpy().astype('float32')
            fake_ratio = (fake_ratio + 1) / 2
            fake_ratio = np.transpose(fake_ratio, (1, 2, 0))

        if opt.ratio == 0:
            fake_output = model.fake_output.cpu().squeeze().numpy().astype('float32')
            fake_output = (fake_output + 1) / 2
            fake_output = np.transpose(fake_output, (1, 2, 0))

        ratio_real = (ratio_real +1)/2
        ratio_low = (ratio_low +1)/2
        input_real = (input_real +1)/2
        output_real = (output_real +1)/2

        ratio_real = np.transpose(ratio_real, (1, 2, 0))
        ratio_low = np.transpose(ratio_low, (1, 2, 0))
        input_real = np.transpose(input_real, (1, 2, 0))
        output_real = np.transpose(output_real, (1, 2, 0))
        # # if opt.direction == 'AtoB':
        # real_output = (2 * (input_real + 1)) / (3 * ratio_real + 1) - 1
        # fake_output = (2 * (input_real + 1)) / (3 * fake_ratio + 1) - 1
        # low_output = (2 * (input_real + 1)) / (3 * ratio_low + 1) - 1
        # else:
        # if opt.ratio == 1:
        real_output = output_real
        # real_otpuut = (2 * (input_real + 1)) / (3 * ratio_real + 1) - 1
        if opt.direction == 'AtoB':
            if opt.ratio == 1:
                fake_output = (2 * (input_real + 1)) / (3 * fake_ratio + 1) - 1
            low_output = (2 * (input_real + 1)) / (3 * ratio_low + 1) - 1
        else:
            if opt.ratio == 1:
                fake_output = (3 * (fake_ratio + input_real * fake_ratio) + input_real - 1) / 2
            low_output = (3 * (ratio_low + input_real * ratio_low) + input_real - 1) / 2
        # fake_output = (3 * (ratio_real + input_real * ratio_real) + input_real - 1) / 2
    # else:
    #     real_output = model.gr_high.cpu().squeeze().numpy().astype('float32')
    #     real_output = (real_output + 1) / 2
    #     real_output = np.transpose(real_output, (1, 2, 0))
    #     low_output = model.input_low.cpu().squeeze().numpy().astype('float32')
    #     low_output = (low_output + 1) / 2
    #     low_output = np.transpose(low_output, (1, 2, 0))
        # fake_output = (3 * (ratio_real + input_real * ratio_real) + input_real - 1) / 2
        # real_output = fake_output
        # low_output = fake_output
        # real_output = (2 * (input_real + 1)) / (3 * ratio_real + 1) - 1
        # B_fake_big_reconstructed = (3 * (B_ratio_big + A * B_ratio_big) + A - 1) / 2
        ratio_low = gama_corect(ratio_low)
        ratio_real = gama_corect(ratio_real)
        real_output = gama_corect(real_output)
        fake_output = gama_corect(fake_output)
        low_output = gama_corect(low_output)
        input_real = gama_corect(input_real)

        ratio_low = (ratio_low * 255).astype('uint8')
        ratio_real = (ratio_real * 255).astype('uint8')
        real_output = (real_output * 255).astype('uint8')
        fake_output = (fake_output * 255).astype('uint8')
        input_real = (input_real * 255).astype('uint8')
        low_output = (low_output * 255).astype('uint8')


        ratio_low = cv2.cvtColor(ratio_low, cv2.COLOR_RGB2BGR)
        ratio_real = cv2.cvtColor(ratio_real, cv2.COLOR_RGB2BGR)
        real_output = cv2.cvtColor(real_output, cv2.COLOR_RGB2BGR)
        fake_output = cv2.cvtColor(fake_output, cv2.COLOR_RGB2BGR)
        input_real = cv2.cvtColor(input_real, cv2.COLOR_RGB2BGR)
        low_output = cv2.cvtColor(low_output, cv2.COLOR_RGB2BGR)


        dir = os.path.join(opt.results_dir, opt.name)
        # cv2.imwrite(dir + '/' + str(i)+"ratio_low.png", ratio_low)
        # cv2.imwrite(dir + '/' + str(i)+"ratio_real.png", ratio_real)
        real_output_name = img_path.replace(".png", "_real_output.png")
        fake_output_name = img_path.replace(".png", "_fake_output.png")
        real_input_name = img_path.replace(".png", "_real_input.png")
        low_output_name = img_path.replace(".png", "_low_output.png")
        if opt.data_mode == "qual":
            real_output_name = img_path+"_real_output.png"
            fake_output_name = img_path+"_fake_output.png"
            real_input_name = img_path+"_real_input.png"
            low_output_name = img_path+"_low_output.png"
        cv2.imwrite(dir + '/' + real_output_name, real_output)
        cv2.imwrite(dir + '/' + fake_output_name, fake_output)
        cv2.imwrite(dir + '/' + real_input_name, input_real)
        cv2.imwrite(dir + '/' + low_output_name, low_output)
