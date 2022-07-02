import os.path
import os
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from PIL import Image
import skimage
import numpy as np
import cv2
import argparse
from PIL.PngImagePlugin import PngImageFile, PngInfo
import random
import torchvision.transforms as transforms
import torch
from skimage.color import rgb2lab


class highResDataset(BaseDataset):

    def __init__(self, opt):

        BaseDataset.__init__(self, opt)
        if opt.phase == 'test':
            if opt.random_test:
                self.fake_data_root = self.opt.fake_data_root

                self.dir_ourdataset = os.path.join(opt.dataroot, 'our_dataset_new_test')
                self.images_dir_ourdataset = sorted(make_dataset(self.dir_ourdataset + '/amb_0.5', 100))

                self.dir_multidataset = os.path.join(opt.dataroot, 'multi_dataset_test_png')
                self.images_dir_multidataset = sorted(make_dataset(self.dir_multidataset + '/amb_0.5/2', 100))

                self.dir_portraitdataset = os.path.join(opt.dataroot, 'portrait_dataset_high_res_test')
                self.images_dir_portraitdataset = sorted(make_dataset(self.dir_portraitdataset + '/amb_0.5/1', 100))

                self.dir_qualdataset = os.path.join(opt.dataroot, 'qual_new')
                self.images_dir_qualdataset = sorted(make_dataset(self.dir_qualdataset, 50))

                self.images_dir_all = self.images_dir_ourdataset + self.images_dir_multidataset + self.images_dir_portraitdataset
            else:
                self.fake_data_root = self.opt.fake_data_root

                self.dir_ourdataset = os.path.join(opt.dataroot, 'our_dataset_new_test')
                self.images_dir_ourdataset = sorted(make_dataset(self.dir_ourdataset, 1000))

                self.dir_multidataset = os.path.join(opt.dataroot, 'multi_dataset_test_png')
                self.images_dir_multidataset = sorted(make_dataset(self.dir_multidataset, 1000))

                self.dir_portraitdataset = os.path.join(opt.dataroot, 'portrait_dataset_high_res_test')
                self.images_dir_portraitdataset = sorted(make_dataset(self.dir_portraitdataset, 1000))

                self.dir_qualdataset = os.path.join(opt.dataroot, 'qual_new')
                self.images_dir_qualdataset = sorted(make_dataset(self.dir_qualdataset, 50))

            if opt.data_mode == 'qual':
                print("qualitative dataset")
                self.images_dir_all = self.images_dir_qualdataset
            else:
                self.images_dir_all = self.images_dir_ourdataset + self.images_dir_multidataset + self.images_dir_portraitdataset

        else:
            self.dir_ourdataset = os.path.join(opt.dataroot, 'our_dataset_new_wb')
            self.images_dir_ourdataset = sorted(make_dataset(self.dir_ourdataset, 100000))
            self.dir_multidataset = os.path.join(opt.dataroot, 'multi_dataset_complete_png_wb')
            self.images_dir_multidataset = sorted(make_dataset(self.dir_multidataset + '/1', 100000))
            self.images_dir_multidataset = self.images_dir_multidataset * 2
            self.dir_portraitdataset = os.path.join(opt.dataroot, 'portrait_dataset_high_res_wb')
            self.images_dir_portraitdataset = sorted(make_dataset(self.dir_portraitdataset + '/1', 100000))
            self.images_dir_portraitdataset = self.images_dir_portraitdataset * 4

            # TO DO: here specify the directory for the generated data
            self.fake_data_root = opt.dataroot
            all_images = self.images_dir_multidataset + self.images_dir_ourdataset + self.images_dir_portraitdataset
            self.images_dir_all = []
            for image in all_images:
                if "flashphoto" in image:
                    self.images_dir_all.append(image)
        self.data_size = opt.load_size
        self.data_root = opt.dataroot
        self.data_size_low = opt.load_size_low
        self.half_fake = opt.half_fake
        self.third_bad = opt.third_bad
        self.all_fake = opt.all_fake
        self.third_blur = opt.third_blur

    def __getitem__(self, index):
        image_path_temp = self.images_dir_all[index]
        image_name = image_path_temp.split('/')[-1]
        image_name = image_name.replace("_flashphoto", "")

        if self.opt.phase == 'test':
            if self.opt.random_test:
                if 'our_dataset_new_test' in image_path_temp:
                    image_path = self.data_root + '/our_dataset_new_test_wb' + '/{}'.format(image_name)
                    fake_path = self.fake_data_root + '/our_dataset_new_test_wb' + '/{}'.format(image_name)
                elif 'multi_dataset_test_png' in image_path_temp:
                    multi_select = random.randint(2, 6)
                    image_path = self.data_root + '/multi_dataset_test_png_wb' + '/{}'.format(
                        multi_select) + '/{}'.format(image_name)
                    fake_path = self.fake_data_root + '/multi_dataset_test_png_wb' + '/{}'.format(
                        multi_select) + '/{}'.format(image_name)
                elif 'portrait_dataset_high_res_test' in image_path_temp:
                    portrait_select = random.randint(1, 7)
                    image_path = self.data_root + '/portrait_dataset_high_res_test_wb' + '/{}'.format(
                        portrait_select) + '/{}'.format(
                        image_name)
                    fake_path = self.fake_data_root + '/portrait_dataset_high_res_test_wb' + '/{}'.format(
                        portrait_select) + '/{}'.format(
                        image_name)
                else:
                    image_path = self.data_root + '/qual_new' + '/{}'.format(
                        image_name)
                    fake_path = self.fake_data_root + '/qual_new' + '/{}'.format(
                        image_name)
            else:
                image_path = image_path_temp
                fake_path = image_path_temp.replace(self.data_root, self.fake_data_root)

        else:
            if 'our_dataset_new_wb' in image_path_temp:
                image_path = self.data_root + '/our_dataset_new_wb' + '/{}'.format(image_name)
                fake_path = self.fake_data_root + '/our_dataset_new_wb''/{}'.format(image_name)
            elif 'multi_dataset_complete_png' in image_path_temp:
                multi_select = random.randint(1, 19)
                image_path = self.data_root + '/multi_dataset_complete_png_wb' + '/{}'.format(
                    multi_select) + '/{}'.format(image_name)
                fake_path = self.fake_data_root + '/multi_dataset_complete_png_wb' + '/{}'.format(
                    multi_select) + '/{}'.format(image_name)
            elif 'portrait_dataset_high_res' in image_path_temp:
                portrait_select = random.randint(1, 20)
                image_path = self.data_root + '/portrait_dataset_high_res_wb' + '/{}'.format(
                    portrait_select) + '/{}'.format(
                    image_name)
                fake_path = self.fake_data_root + '/portrait_dataset_high_res_wb' + '/{}'.format(
                    portrait_select) + '/{}'.format(
                    image_name)

        ambient = Image.open(image_path.replace(".png", "_ambient.png"))
        flash = Image.open(image_path.replace(".png", "_flash.png"))
        ambient_float = skimage.img_as_float(ambient)
        flash_float = skimage.img_as_float(flash)
        
        # NOTE (chris): I don't think this is added to the opts
        # # normalize ambient
        # if self.opt.normalize_ambient == 1:
        #     ambient_brightness = self.getBrightness(ambient_float)
        #     ambient_float = ambient_float * norm_amb / ambient_brightness
        #     ambient_float[ambient_float < 0] = 0
        #     ambient_float[ambient_float > 1] = 1
        # # normalize flash
        # if self.opt.normalize_flash == 1:
        #     flash_brightness = self.getBrightness(flash_float)
        #     flash_float = flash_float * norm_flash / flash_brightness
        #     flash_float[flash_float < 0] = 0
        #     flash_float[flash_float > 1] = 1

        # compute flashPhoto
        # flash photo = whitebalanced flash + colored ambient
        flashPhoto_float = flash_float + ambient_float
        flashPhoto_float[flashPhoto_float < 0] = 0
        flashPhoto_float[flashPhoto_float > 1] = 1
        badFlashPhoto = 0.7 * flash_float + ambient_float
        badAmbient = 0.3 * flash_float + ambient_float

        badAmbient = Image.fromarray((badAmbient * 255).astype('uint8'))
        badFlashPhoto = Image.fromarray((badFlashPhoto * 255).astype('uint8'))
        ambient = Image.fromarray((ambient_float * 255).astype('uint8'))
        flashPhoto = Image.fromarray((flashPhoto_float * 255).astype('uint8'))

        ambient_highRes = ambient.resize((self.data_size, self.data_size))
        flashPhoto_highRes = flashPhoto.resize((self.data_size, self.data_size))

        ambient_lowRes = ambient.resize((self.data_size_low, self.data_size_low))
        flashPhoto_lowRes = flashPhoto.resize((self.data_size_low, self.data_size_low))

        ambient_lowRes = ambient_lowRes.resize((self.data_size, self.data_size))
        flashPhoto_lowRes = flashPhoto_lowRes.resize((self.data_size, self.data_size))

        badAmbient = badAmbient.resize((self.data_size_low, self.data_size_low))
        badFlashPhoto = badFlashPhoto.resize((self.data_size_low, self.data_size_low))

        badAmbient = badAmbient.resize((self.data_size, self.data_size))
        badFlashPhoto = badFlashPhoto.resize((self.data_size, self.data_size))

        ambient_highRes_fl = skimage.img_as_float(ambient_highRes)
        flashPhoto_highRes_fl = skimage.img_as_float(flashPhoto_highRes)

        badAmbient_fl = skimage.img_as_float(badAmbient)
        badFlashPhoto_fl = skimage.img_as_float(badFlashPhoto)

        ambient_lowRes_fl = skimage.img_as_float(ambient_lowRes)
        flashPhoto_lowRes_fl = skimage.img_as_float(flashPhoto_lowRes)

        if self.opt.data_mode == "qual":
            fake_ambient_path = fake_path + "_fake_B.png"
            fake_flashPhoto_path = fake_path + "_fake_A.png"
        else:
            fake_ambient_path = fake_path.replace(".png", "_fake_ambient_dec.png")
            fake_flashPhoto_path = fake_path.replace(".png", "_fake_flashPhoto_gen.png")

        fake_ambient = Image.open(fake_ambient_path)
        fake_flashPhoto = Image.open(fake_flashPhoto_path)

        fake_ambient = fake_ambient.resize((self.data_size, self.data_size))
        fake_flashPhoto = fake_flashPhoto.resize((self.data_size, self.data_size))

        fake_ambient_fl = skimage.img_as_float(fake_ambient)
        fake_flashPhoto_fl = skimage.img_as_float(fake_flashPhoto)
        
        # NOTE (chris): I wrote the results out in linear space
        # fake_ambient_fl = self.lin(fake_ambient_fl)
        # fake_flashPhoto_fl = self.lin(fake_flashPhoto_fl)

        if self.third_blur or self.opt.all_blur:
            if self.third_blur:
                blur = random.randint(1, 3)
            else:
                blur = 1
            if blur == 1:
                ambient_lowRes_fl = self.blur(ambient_lowRes_fl, full_size=self.opt.full_size_blur)
                badAmbient_fl = self.blur(badAmbient_fl, full_size=self.opt.full_size_blur)
                badFlashPhoto_fl = self.blur(badFlashPhoto_fl, full_size=self.opt.full_size_blur)
                flashPhoto_lowRes_fl = self.blur(flashPhoto_lowRes_fl, full_size=self.opt.full_size_blur)
                fake_flashPhoto_fl = self.blur(fake_flashPhoto_fl, full_size=self.opt.full_size_blur)
                fake_ambient_fl = self.blur(fake_ambient_fl, full_size=self.opt.full_size_blur)

        A_ratio_high = (2 * (ambient_highRes_fl + 1) / (3 * (flashPhoto_highRes_fl + 1))) - 1 / 3
        B_ratio_high = (2 * (ambient_highRes_fl + 1) / (3 * (flashPhoto_highRes_fl + 1))) - 1 / 3
        A_ratio_low = (2 * (ambient_highRes_fl + 1) / (3 * (flashPhoto_lowRes_fl + 1))) - 1 / 3
        B_ratio_low = (2 * (ambient_lowRes_fl + 1) / (3 * (flashPhoto_highRes_fl + 1))) - 1 / 3
        A_ratio_bad = (2 * (ambient_highRes_fl + 1) / (3 * (badFlashPhoto_fl + 1))) - 1 / 3
        B_ratio_bad = (2 * (ambient_lowRes_fl + 1) / (3 * (badAmbient_fl + 1))) - 1 / 3
        A_ratio_fake = (2 * (ambient_highRes_fl + 1) / (3 * (fake_flashPhoto_fl + 1))) - 1 / 3
        B_ratio_fake = (2 * (fake_ambient_fl + 1) / (3 * (flashPhoto_highRes_fl + 1))) - 1 / 3

        A_ratio_fake = Image.fromarray((A_ratio_fake * 255).astype('uint8'))
        B_ratio_fake = Image.fromarray((B_ratio_fake * 255).astype('uint8'))
        A_ratio_high = Image.fromarray((A_ratio_high * 255).astype('uint8'))
        B_ratio_high = Image.fromarray((B_ratio_high * 255).astype('uint8'))
        A_ratio_low = Image.fromarray((A_ratio_low * 255).astype('uint8'))
        B_ratio_low = Image.fromarray((B_ratio_low * 255).astype('uint8'))
        A_ratio_bad = Image.fromarray((A_ratio_bad * 255).astype('uint8'))
        B_ratio_bad = Image.fromarray((B_ratio_bad * 255).astype('uint8'))

        transform_params = get_params(self.opt, A_ratio_high.size)
        rgb_transform = get_transform(self.opt, transform_params, grayscale=False)

        A_ratio_high = rgb_transform(A_ratio_high)
        B_ratio_high = rgb_transform(B_ratio_high)

        A_ratio_low = rgb_transform(A_ratio_low)
        B_ratio_low = rgb_transform(B_ratio_low)

        flashPhoto_highRes = rgb_transform(flashPhoto_highRes)
        ambient_highRes = rgb_transform(ambient_highRes)

        ambient_lowRes = rgb_transform(ambient_lowRes)
        flashPhoto_lowRes = rgb_transform(flashPhoto_lowRes)

        # if self.opt.phase == 'train':
        A_ratio_fake = rgb_transform(A_ratio_fake)
        B_ratio_fake = rgb_transform(B_ratio_fake)

        A_ratio_bad = rgb_transform(A_ratio_bad)
        B_ratio_bad = rgb_transform(B_ratio_bad)
        if self.half_fake:
            fake_or_low = random.randint(1, 2)
            if fake_or_low == 1:
                A_ratio_low = A_ratio_fake
                B_ratio_low = B_ratio_fake
        if self.all_fake or self.opt.phase == 'test':
            A_ratio_low = A_ratio_fake
            B_ratio_low = B_ratio_fake
        if self.third_bad:
            bad_or_low = random.randint(1, 3)
            if bad_or_low == 1:
                A_ratio_low = A_ratio_bad
                B_ratio_low = B_ratio_bad

        return {'ambient_lowRes': ambient_lowRes, 'flashPhoto_lowRes': flashPhoto_lowRes, 'A_ratio_low': A_ratio_low,
                'A_ratio_high': A_ratio_high, 'B_ratio_high': B_ratio_high, 'B_ratio_low': B_ratio_low,
                'ambient_highRes': ambient_highRes, 'flashPhoto_highRes': flashPhoto_highRes,
                'A_paths': image_path, 'B_paths': image_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.images_dir_all)

    def blur(self, image, radius=200, full_size=False):
        blurred_image = cv2.GaussianBlur(image, (21, 21), 0)
        if full_size:
            return blurred_image
        else:
            width, height, c = image.shape
            mask = np.zeros((width, height, 3), dtype=np.uint8)
            rand_x = random.randrange(radius + 1, width - radius - 1)
            rand_y = random.randrange(radius + 1, height - radius - 1)
            mask = cv2.circle(mask, (rand_x, rand_y), 200, (255, 255, 255), -1)
            image = np.where(mask == (255, 255, 255), blurred_image, image)
            return image

    def getBrightness(self, image):
        lab = rgb2lab(image)
        w = len(image)
        h = len(image[0])
        L = lab[:, :, 0]
        L = np.array(L)
        L_flat = np.reshape(L, (w * h))
        L_flat = np.sort(L_flat)
        len_all = len(L_flat)
        leave_out = int(len(L_flat) / 10)
        brightness = sum(L_flat[leave_out:-leave_out])
        brightness = brightness / (len_all * 8 / 10)
        return brightness

    def gama_corect(self, rgb):
        srgb = np.zeros_like(rgb)
        mask1 = (rgb > 0) * (rgb < 0.0031308)
        mask2 = (1 - mask1).astype(bool)
        srgb[mask1] = 12.92 * rgb[mask1]
        srgb[mask2] = 1.055 * np.power(rgb[mask2], 0.41666) - 0.055
        srgb[srgb < 0] = 0
        return srgb

    def lin(self, srgb):
        srgb = srgb.astype(np.float)
        rgb = np.zeros_like(srgb).astype(np.float)
        srgb = srgb
        mask1 = srgb <= 0.04045
        mask2 = (1 - mask1).astype(bool)
        rgb[mask1] = srgb[mask1] / 12.92
        rgb[mask2] = ((srgb[mask2] + 0.055) / 1.055) ** 2.4
        rgb = rgb
        return rgb
