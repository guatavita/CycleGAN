# Created by Bastien Rigaud at 18/02/2022
# Bastien Rigaud, PhD
# Laboratoire Traitement du Signal et de l'Image (LTSI), INSERM U1099
# Campus de Beaulieu, Université de Rennes 1
# 35042 Rennes, FRANCE
# bastien.rigaud@univ-rennes1.fr
# Description:


# divers
import os
import sys
import getopt
import argparse
import time
import glob


# mandatory for linux, otherwise bool are converted to string
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(description="Run training")
parser.add_argument("--tb_path", default=r'C:\Data\DELPEL\results\Tensorboard_cGAN', type=str, help="tensorboard folder")
parser.add_argument("--base_path", default=r'C:\Data\DELPEL\results\img_label_data\CBCT\tfrecords', type=str, help="tfrecord base folder for train/validation folders")
parser.add_argument("--model_desc", default="CycleGAN", type=str, help="model name")
parser.add_argument("--loss_func1", default='mse', choices=['cc', 'mse'], type=str, help="loss function")
parser.add_argument("--loss_func2", default='mse', choices=['cc', 'mse'], type=str, help="loss function")
parser.add_argument("--optimizer", default="adam", choices=['adam', 'sgd', 'sgdn', 'both'], type=str, help="optimizer function")
parser.add_argument("--batch_size", default=1, type=int, help="batch size for training")
parser.add_argument("--lr", default=0.0001, type=float, help="stable learning rate")
parser.add_argument("--per_img_std", default=True, type=str2bool, help="bool to run per image centered normalization")
parser.add_argument("--max_noise", default=0.0, type=float, help="gaussian noise augmentation")
parser.add_argument("--scale_aug", default=1.0, type=float, help="scale value to run scaling augmentation")
parser.add_argument("--crop_aug", default=False, type=str2bool, help="boolean flag to run croping augmentation")
parser.add_argument("--lr_flip_aug", default=False, type=str2bool, help="boolean flag to run left right flip augmentation")
parser.add_argument("--ud_flip_aug", default=False, type=str2bool, help="boolean flag to run up down flip augmentation")
parser.add_argument("--contrast_aug", default=0.0, type=float, help="factor to run random contrast variation augmentation between 1-factor and 1+factor")
parser.add_argument("--rotation_angle_aug", default=0.0, type=float, help="rotation angle value (radians) to run rotation augmentation")
parser.add_argument("--translation_aug", default=0.0, type=float, help="translation value (x, y) to run translation augmentation")
parser.add_argument("--img_size", default=512, type=int, help="row and col for input image size")
parser.add_argument("--gpu", type=str, default="", help="select GPU id")
parser.add_argument("--iteration", default=1, type=int, help="iteration id to run multiple times the same hparams")
args = parser.parse_args()

# GPU device
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = 'true'
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

# tensorflow
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from tensorboard.plugins.hparams.keras import Callback

# tensorflow addons
import tensorflow_addons as tfa

# Brian's Utils
from Base_Deeplearning_Code.Data_Generators.Image_Processors_Module.src.Processors.TFDataSetProcessors import *
from Base_Deeplearning_Code.Callbacks.TF2_Callbacks import Add_Images_and_LR, SparseCategoricalMeanDSC
from Base_Deeplearning_Code.Cyclical_Learning_Rate.clr_callback_TF2 import CyclicLR
from Base_Deeplearning_Code.Plot_And_Scroll_Images.Plot_Scroll_Images import plot_scroll_Image, \
    plot_Image_Scroll_Bar_Image
from Base_Deeplearning_Code.Finding_Optimization_Parameters.Hyper_Parameter_Functions import determine_if_in_excel, \
    return_hparams

# network
from networks.UNet3D import *
from return_generator import *



def main():


if __name__ == '__main__':
    main()
