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
parser.add_argument("--tb_path", default=r'/workspace/Tensorboard/Tensorboard_cGAN', type=str,
                    help="tensorboard folder")
parser.add_argument("--base_path", default=r'/workspace/Data/unpaired_tfrecords', type=str,
                    help="tfrecord base folder for train/validation folders")
parser.add_argument("--model_desc", default="CycleGAN", type=str, help="model name")
parser.add_argument("--optimizer", default="adam", choices=['adam', 'sgd', 'sgdn', 'both'], type=str,
                    help="optimizer function")
parser.add_argument("--normalization", default="instance", choices=["batch", "group", "instance"], type=str,
                    help="normalization function")
parser.add_argument("--epoch", default=200, type=int, help="max number of epochs")
parser.add_argument("--batch_size", default=1, type=int, help="batch size for training")
parser.add_argument("--lr", default=0.0002, type=float, help="stable learning rate")
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
parser.add_argument("--gpu", type=str, default="1", help="select GPU id")
parser.add_argument("--iteration", default=3, type=int, help="iteration id to run multiple times the same hparams")
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
from PlotScrollNumpyArrays.Plot_Scroll_Images import plot_scroll_Image

from Network.CycleGAN import CycleGAN
from return_generator import *
from Callback.cycle_images_callback import Add_Cycle_Images


def create_hparams_data(model_desc, tensorboard_path, batch_size, lr, epoch, optimizer, normalization,
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


# TODO (cf https://www.kaggle.com/dimitreoliveira/improving-cyclegan-monet-paintings/notebook)
# Transformer with residual blocks [++]
# Residual connections between Generator and Discriminator [++]
# Not using InstanceNorm at the first layer of both generator and discriminator [++]
# Better InstanceNorm layer initialization [++]
# Residual connection with Concatenate instead of Add [+]
# Data augmentations (flips, rotations, and crops) [+]
# Discriminator with label smoothing [+]

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

    # hparams, trial_id = None, '9999'
    hparams, trial_id = create_hparams_data(model_desc, tensorboard_path, batch_size, lr, epoch, optimizer,
                                            normalization, max_noise, scale_aug, crop_aug, lr_flip_aug, ud_flip_aug,
                                            rotation_angle_aug, img_size, translation_aug, per_img_std, contrast_aug,
                                            iteration)

    debug = False
    if debug:
        prep_tfrecord_cache = False
        shuffle = False
    else:
        prep_tfrecord_cache = True
        shuffle = True

    image_generator_args = {'base_path': base_path, 'image_keys': ('image',), 'interp_keys': ('bilinear',),
                            'filling_keys': ('constant',), 'dtype_keys': ('float16',), 'model_desc': model_desc,
                            'debug': debug, 'max_noise': max_noise, 'scale_aug': scale_aug, 'crop_aug': crop_aug,
                            'lr_flip_aug': lr_flip_aug, 'ud_flip_aug': ud_flip_aug,
                            'rotation_angle_aug': rotation_angle_aug, 'translation_aug': translation_aug,
                            'per_img_std': per_img_std, 'contrast_aug': contrast_aug, 'shuffle': shuffle,
                            'prep_tfrecord_cache': prep_tfrecord_cache, 'ct_clip': True}
    annotation_generator_args = {'base_path': base_path, 'image_keys': ('annotation',), 'interp_keys': ('nearest',),
                                 'filling_keys': ('constant',), 'dtype_keys': ('float16',), 'model_desc': model_desc,
                                 'debug': debug, 'max_noise': 0.0, 'scale_aug': scale_aug, 'crop_aug': crop_aug,
                                 'lr_flip_aug': lr_flip_aug, 'ud_flip_aug': ud_flip_aug,
                                 'rotation_angle_aug': rotation_angle_aug, 'translation_aug': translation_aug,
                                 'per_img_std': False, 'contrast_aug': contrast_aug, 'shuffle': shuffle,
                                 'prep_tfrecord_cache': prep_tfrecord_cache, 'ct_clip': False}

    train_generator_X, train_generator_Y = return_generator(is_validation=False, batch_size=batch_size,
                                                            **image_generator_args), \
                                           return_generator(is_validation=False, batch_size=batch_size,
                                                            **annotation_generator_args)
    validation_generator_X, validation_generator_Y = return_generator(is_validation=True, batch_size=1,
                                                                      **image_generator_args), \
                                                     return_generator(is_validation=True, batch_size=1,
                                                                      **annotation_generator_args)

    # train_gen_x = train_generator_X.data_set.as_numpy_iterator()
    # x = next(train_gen_x)
    # train_gen_y = train_generator_Y.data_set.as_numpy_iterator()
    # y = next(train_gen_y)
    # train_generators = tf.data.Dataset.zip((train_generator_X.data_set, train_generator_Y.data_set)).as_numpy_iterator()
    # x, y = next(train_generators)
    # plot_scroll_Image(x[0][...,0])
    # plot_scroll_Image(y[0][...,0])

    # Loss function for evaluating adversarial loss
    adv_loss_fn = tf.keras.losses.MeanSquaredError()

    # Define the loss function for the generators
    def generator_loss_fn(fake):
        fake_loss = adv_loss_fn(tf.ones_like(fake), fake)
        return fake_loss

    # Define the loss function for the discriminators
    def discriminator_loss_fn(real, fake):
        real_loss = adv_loss_fn(tf.ones_like(real), real)
        fake_loss = adv_loss_fn(tf.zeros_like(fake), fake)
        return (real_loss + fake_loss) * 0.5

    # Create cycle gan model
    cycle_gan_model = CycleGAN(input_shape=(img_size, img_size, 1))

    # Compile the model
    cycle_gan_model.compile(
        gen_G_optimizer=tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.5),
        gen_F_optimizer=tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.5),
        disc_X_optimizer=tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.5),
        disc_Y_optimizer=tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.5),
        gen_loss_fn=generator_loss_fn,
        disc_loss_fn=discriminator_loss_fn,
    )

    # callbacks
    tensorboard_output = os.path.join(tensorboard_path, 'Trial_ID_{}'.format(trial_id))
    if not os.path.exists(tensorboard_output):
        os.makedirs(tensorboard_output)

    tensorboard = TensorBoard(log_dir=tensorboard_output, write_graph=False, write_grads=False, write_images=False,
                              update_freq='epoch', histogram_freq=5, profile_batch=0)

    checkpoint_path = os.path.join(tensorboard_path, 'Trial_ID_{}'.format(trial_id), model_desc + '.hdf5')

    checkpoint = ModelCheckpoint(checkpoint_path, save_weights_only=True, save_freq='epoch', verbose=1)

    tensorboard_img = Add_Cycle_Images(tensorboard_output,
                                       validation_data=tf.data.Dataset.zip((validation_generator_X.data_set,
                                                                            validation_generator_Y.data_set)),
                                       number_of_images=5, image_rows=img_size//2, image_cols=img_size//2,
                                       frequency=1)

    callbacks = [checkpoint, tensorboard, tensorboard_img]

    if hparams is not None:
        hp_callback = Callback(tensorboard_output, hparams=hparams, trial_id='Trial_ID:{}'.format(trial_id))
        callbacks += [hp_callback]

    print('Model created at: ' + os.path.abspath(tensorboard_output))
    print("-------------- Running model")
    print("Number of training steps: {}, {}".format(len(train_generator_X), len(train_generator_Y)))
    print("Number of validation steps: {}, {}".format(len(validation_generator_X), len(validation_generator_Y)))
    print('This is the number of trainable weights:', len(cycle_gan_model.trainable_weights))

    cycle_gan_model.fit(tf.data.Dataset.zip((train_generator_X.data_set, train_generator_Y.data_set)), epochs=epoch,
              steps_per_epoch=min(len(train_generator_X), len(train_generator_Y)),
              validation_data=tf.data.Dataset.zip((validation_generator_X.data_set, validation_generator_Y.data_set)),
              validation_steps=min(len(validation_generator_X), len(validation_generator_Y)),
              callbacks=callbacks, verbose=1, use_multiprocessing=False, workers=1, max_queue_size=10)


if __name__ == '__main__':
    main()
