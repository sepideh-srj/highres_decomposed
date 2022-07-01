import shutil
import os
import os
from PIL import Image
import numpy
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse
from skimage.metrics import  peak_signal_noise_ratio as psnr
from cv2 import cv2
import argparse
parser = argparse.ArgumentParser(description='--dir')
parser.add_argument('--dir', type=str,required=True)

args = parser.parse_args()
dirName = args.dir
realDir = dirName+ "/real"
fakeHighDir = dirName + "/highFake"
fakeLowDir = dirName + "/lowFake"
print(realDir)
if not os.path.exists(realDir):
    os.makedirs(realDir)
    os.makedirs(fakeHighDir)
    os.makedirs(fakeLowDir)

images = os.listdir(dirName)


for image in images:
    if 'fake_output' in image:
        original = os.path.join(dirName, image)
        newDir =dirName + "/highFake"
        target = os.path.join(newDir, image)
        shutil.copyfile(original, target)
    elif 'low_output' in image:
        original = os.path.join(dirName, image)
        newDir =dirName + "/lowFake"
        target = os.path.join(newDir, image)
        shutil.copyfile(original, target)
    elif 'real_output' in image:
        original = os.path.join(dirName, image)
        newDir =dirName + "/real"
        target = os.path.join(newDir, image)
        shutil.copyfile(original, target)


# for image in images:
#     if 'real_A' in image:
#         if 'portrait' in image:
#             original = os.path.join(dirName, image)
#             target = original.replace('images', 'real_A_por')
#             shutil.copyfile(original, target)
#         if 'multi' in image:
#             original = os.path.join(dirName, image)
#             target = original.replace('images', 'real_A_multi')
#             shutil.copyfile(original, target)
#         if 'our' in image:
#             original = os.path.join(dirName, image)
#             target = original.replace('images', 'real_A_our')
#             shutil.copyfile(original, target)
#     elif 'real_B' in image:
#         if 'portrait' in image:
#             original = os.path.join(dirName, image)
#             target = original.replace('images', 'real_B_por')
#             shutil.copyfile(original, target)
#         if 'multi' in image:
#             original = os.path.join(dirName, image)
#             target = original.replace('images', 'real_B_multi')
#             shutil.copyfile(original, target)
#         if 'our' in image:
#             original = os.path.join(dirName, image)
#             target = original.replace('images', 'real_B_our')
#             shutil.copyfile(original, target)
#
#
#     elif 'fake_A' in image:
#         if 'portrait' in image:
#             original = os.path.join(dirName, image)
#             target = original.replace('images', 'fake_A_por')
#             shutil.copyfile(original, target)
#         if 'multi' in image:
#             original = os.path.join(dirName, image)
#             target = original.replace('images', 'fake_A_multi')
#             shutil.copyfile(original, target)
#         if 'our' in image:
#             original = os.path.join(dirName, image)
#             target = original.replace('images', 'fake_A_our')
#             shutil.copyfile(original, target)
#
#
#     elif 'fake_B' in image:
#         if 'portrait' in image:
#             original = os.path.join(dirName, image)
#             target = original.replace('images', 'fake_B_por')
#             shutil.copyfile(original, target)
#         if 'multi' in image:
#             original = os.path.join(dirName, image)
#             target = original.replace('images', 'fake_B_multi')
#             shutil.copyfile(original, target)
#         if 'our' in image:
#             original = os.path.join(dirName, image)
#             target = original.replace('images', 'fake_B_our')
#             shutil.copyfile(original, target)


# import shutil
# import os
# import os
# from PIL import Image
# import numpy
# from skimage.metrics import structural_similarity as ssim
# from skimage.metrics import mean_squared_error as mse
# from skimage.metrics import  peak_signal_noise_ratio as psnr
# from cv2 import cv2
# import argparse
# parser = argparse.ArgumentParser(description='--dir')
# parser.add_argument('--dir', type=str,required=True)
#
# args = parser.parse_args()
# dirName = args.dir
# realADir = dirName.replace('images', 'real')
# # realBDir = dirName.replace('images', 'real_B')
# fakeADir = dirName.replace('images', 'fake')
#
# if not os.path.exists(realADir):
#     os.makedirs(realADir)
#     # os.makedirs(realBDir)
#     os.makedirs(fakeADir)
#     # os.makedirs(fakeBDir)
#
# images = os.listdir(dirName)
#
#
# for image in images:
#     # if 'real_A' in image:
#     #     original = os.path.join(dirName, image)
#     #     target = original.replace('images', 'real_A')
#     #     shutil.copyfile(original, target)
#     if 'real_B' in image:
#         original = os.path.join(dirName, image)
#         target = original.replace('images', 'real')
#         shutil.copyfile(original, target)
#     # elif 'fake_A' in image:
#     #     original = os.path.join(dirName, image)
#     #     target = original.replace('images', 'fake_A')
#     #     shutil.copyfile(original, target)
#     elif 'fake_B' in image:
#         original = os.path.join(dirName, image)
#         target = original.replace('images', 'fake')
#         shutil.copyfile(original, target)

