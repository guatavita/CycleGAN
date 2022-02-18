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
parser.add_argument("--tb_path", default=r'C:\Data\DELPEL\results\Tensorboard_cGAN', type=str,
                    help="tensorboard folder")
parser.add_argument("--base_path", default=r'C:\Data\DELPEL\results\img_label_data\CBCT\tfrecords', type=str,
                    help="tfrecord base folder for train/validation folders")
parser.add_argument("--model_desc", default="CycleGAN", type=str, help="model name")
parser.add_argument("--loss_func1", default='mse', choices=['cc', 'mse'], type=str, help="loss function")
parser.add_argument("--loss_func2", default='mse', choices=['cc', 'mse'], type=str, help="loss function")
parser.add_argument("--optimizer", default="adam", choices=['adam', 'sgd', 'sgdn', 'both'], type=str,
                    help="optimizer function")
parser.add_argument("--normalization", default="instance", choices=["batch", "group", "instance"], type=str,
                    help="normalization function")
parser.add_argument("--batch_size", default=1, type=int, help="batch size for training")
parser.add_argument("--lr", default=0.0001, type=float, help="stable learning rate")
parser.add_argument("--per_img_std", default=True, type=str2bool, help="bool to run per image centered normalization")
parser.add_argument("--max_noise", default=0.0, type=float, help="gaussian noise augmentation")
parser.add_argument("--scale_aug", default=1.0, type=float, help="scale value to run scaling augmentation")
parser.add_argument("--crop_aug", default=False, type=str2bool, help="boolean flag to run croping augmentation")
parser.add_argument("--lr_flip_aug", default=False, type=str2bool,
                    help="boolean flag to run left right flip augmentation")
parser.add_argument("--ud_flip_aug", default=False, type=str2bool, help="boolean flag to run up down flip augmentation")
parser.add_argument("--contrast_aug", default=0.0, type=float,
                    help="factor to run random contrast variation augmentation between 1-factor and 1+factor")
parser.add_argument("--rotation_angle_aug", default=0.0, type=float,
                    help="rotation angle value (radians) to run rotation augmentation")
parser.add_argument("--translation_aug", default=0.0, type=float,
                    help="translation value (x, y) to run translation augmentation")
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
from tensorboard.plugins.hparams._keras import Callback

# tensorflow addons
import tensorflow_addons as tfa

# Brian's Utils
from Base_Deeplearning_Code.Data_Generators.Image_Processors_Module.src.Processors.TFDataSetProcessors import *
from Base_Deeplearning_Code.Finding_Optimization_Parameters.Hyper_Parameter_Functions import determine_if_in_excel, \
    return_hparams

# network
from Network.CycleGAN import CycleGAN
from return_generator import *


def create_hparams_data(model_desc, tensorboard_path, batch_size, lr, epoch, loss_function, optimizer, normalization,
                        max_noise, scale_aug, crop_aug, lr_flip_aug, ud_flip_aug, rotation_angle_aug, img_size,
                        translation_aug, per_img_std, contrast_aug, iteration):
    if not os.path.exists(tensorboard_path):
        os.makedirs(tensorboard_path)
    excel_path = os.path.join(tensorboard_path, 'parameters_list_by_trial_id.xlsx')

    run_data = {'model_desc': model_desc,
                'tensorboard_path': tensorboard_path,
                'batch_size': batch_size,
                'lr': lr,
                'epoch': epoch,
                'loss_function': loss_function,
                'optimizer': optimizer,
                'normalization': normalization,
                'max_noise': max_noise,
                'scale_aug': scale_aug,
                'crop_aug': crop_aug,
                'lr_flip_aug': lr_flip_aug,
                'ud_flip_aug': ud_flip_aug,
                'rotation_angle_aug': rotation_angle_aug,
                'img_size': img_size,
                'translation_aug': translation_aug,
                'per_img_std': per_img_std,
                'contrast_aug': contrast_aug,
                'iteration': iteration}

    run_data['Trial_ID'] = 0
    features_list = list(run_data.keys())

    # this does not work if when field is empty
    if determine_if_in_excel(excel_path, run_data, features_list=features_list):
        print(" WARNING model has been run already...")
        sys.exit()
    trial_id = run_data['Trial_ID']
    hparams = return_hparams(run_data, features_list=features_list, excluded_keys=[])

    return hparams, trial_id


def main():
    print("List of arguments:")
    for arg in vars(args):
        print('  {:20}: {:>20}'.format(arg, getattr(args, arg) or "0.0/None/False"))

    model_desc = args.model_desc
    tensorboard_path = args.tb_path
    base_path = args.base_path
    batch_size = args.batch_size
    lr = args.lr
    epoch = args.epoch
    loss_function = args.loss_func
    optimizer = args.optimizer
    normalization = args.normalization
    per_img_std = args.per_img_std
    max_noise = args.max_noise
    scale_aug = args.scale_aug
    crop_aug = args.crop_aug
    lr_flip_aug = args.lr_flip_aug
    ud_flip_aug = args.ud_flip_aug
    contrast_aug = args.contrast_aug
    rotation_angle_aug = args.rotation_angle_aug
    translation_aug = args.translation_aug
    img_size = args.img_size
    iteration = args.iteration

    hparams, trial_id = None, '9999'
    # hparams, trial_id = create_hparams_data(model_desc, tensorboard_path, batch_size, lr, epoch, loss_function,
    #                                         optimizer, normalization, max_noise, scale_aug, crop_aug, lr_flip_aug,
    #                                         ud_flip_aug, rotation_angle_aug, img_size, translation_aug, per_img_std,
    #                                         contrast_aug, iteration)

    image_generator_args = {'base_path': base_path, 'image_keys': ('image',), 'interp_keys': ('bilinear',),
                            'filling_keys': ('constant',), 'dtype_keys': ('float16',), 'model_desc': model_desc,
                            'debug': False, 'max_noise': max_noise, 'scale_aug': scale_aug, 'crop_aug': crop_aug,
                            'lr_flip_aug': lr_flip_aug, 'ud_flip_aug': ud_flip_aug,
                            'rotation_angle_aug': rotation_angle_aug, 'translation_aug': translation_aug,
                            'per_img_std': per_img_std, 'contrast_aug': contrast_aug, 'shuffle': True,
                            'prep_tfrecord_cache': True, 'ct_clip': True}
    annotation_generator_args = {'base_path': base_path, 'image_keys': ('annotation',), 'interp_keys': ('nearest',),
                                 'filling_keys': ('constant',), 'dtype_keys': ('float16',), 'model_desc': model_desc,
                                 'debug': False, 'max_noise': 0.0, 'scale_aug': scale_aug, 'crop_aug': crop_aug,
                                 'lr_flip_aug': lr_flip_aug, 'ud_flip_aug': ud_flip_aug,
                                 'rotation_angle_aug': rotation_angle_aug, 'translation_aug': translation_aug,
                                 'per_img_std': False, 'contrast_aug': contrast_aug, 'shuffle': True,
                                 'prep_tfrecord_cache': True, 'ct_clip': False}

    train_generator_X, train_generator_Y = return_generator(is_validation=False, batch_size=batch_size,
                                                            **image_generator_args), \
                                           return_generator(is_validation=False, batch_size=batch_size,
                                                            **annotation_generator_args)
    validation_generator_X, validation_generator_Y = return_generator(is_validation=True, batch_size=1,
                                                                      **image_generator_args), \
                                                     return_generator(is_validation=True, batch_size=1,
                                                                      **annotation_generator_args)


if __name__ == '__main__':
    main()
