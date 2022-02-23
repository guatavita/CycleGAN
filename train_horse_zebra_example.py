# Created by Bastien Rigaud at 17/02/2022
# Bastien Rigaud, PhD
# Laboratoire Traitement du Signal et de l'Image (LTSI), INSERM U1099
# Campus de Beaulieu, Université de Rennes 1
# 35042 Rennes, FRANCE
# bastien.rigaud@univ-rennes1.fr
# Description:

import os

# GPU device
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = 'true'
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import tensorflow_addons as tfa
import tensorflow_datasets as tfds

autotune = tf.data.AUTOTUNE

from Network.CycleGAN import CycleGAN

# Define the standard image size.
orig_img_size = (286, 286)
# Size of the random crops to be used during training.
input_img_size = (256, 256, 3)

# Loss function for evaluating adversarial loss
adv_loss_fn = keras.losses.MeanSquaredError()


# Define the loss function for the generators
def generator_loss_fn(fake):
    fake_loss = adv_loss_fn(tf.ones_like(fake), fake)
    return fake_loss


# Define the loss function for the discriminators
def discriminator_loss_fn(real, fake):
    real_loss = adv_loss_fn(tf.ones_like(real), real)
    fake_loss = adv_loss_fn(tf.zeros_like(fake), fake)
    return (real_loss + fake_loss) * 0.5


class GANMonitor(keras.callbacks.Callback):
    """A callback to generate and save images after each epoch"""

    def __init__(self, test_dataset, num_img=4):
        self.num_img = num_img
        self.test_dataset = test_dataset

    def on_epoch_end(self, epoch, logs=None):
        _, ax = plt.subplots(4, 2, figsize=(12, 12))
        for i, img in enumerate(self.test_dataset.take(self.num_img)):
            prediction = self.model.gen_G(img)[0].numpy()
            prediction = (prediction * 127.5 + 127.5).astype(np.uint8)
            img = (img[0] * 127.5 + 127.5).numpy().astype(np.uint8)

            ax[i, 0].imshow(img)
            ax[i, 1].imshow(prediction)
            ax[i, 0].set_title("Input image")
            ax[i, 1].set_title("Translated image")
            ax[i, 0].axis("off")
            ax[i, 1].axis("off")

            prediction = keras.preprocessing.image.array_to_img(prediction)
            prediction.save(r"C:\Bastien\Python\CycleGAN\output\generated_img_{i}_{epoch}.png".format(i=i, epoch=epoch + 1))
        plt.show()
        plt.close()


def normalize_img(img):
    img = tf.cast(img, dtype=tf.float32)
    # Map values in the range [-1, 1]
    return (img / 127.5) - 1.0


def preprocess_train_image(img, label):
    # Random flip
    img = tf.image.random_flip_left_right(img)
    # Resize to the original size first
    img = tf.image.resize(img, [*orig_img_size])
    # Random crop to 256X256
    img = tf.image.random_crop(img, size=[*input_img_size])
    # Normalize the pixel values in the range [-1, 1]
    img = normalize_img(img)
    return img


def preprocess_test_image(img, label):
    # Only resizing and normalization for the test images.
    img = tf.image.resize(img, [input_img_size[0], input_img_size[1]])
    img = normalize_img(img)
    return img


def main():
    # Load the horse-zebra dataset using tensorflow-datasets.
    dataset, _ = tfds.load('cycle_gan/horse2zebra', with_info=True, as_supervised=True)
    train_horses, train_zebras = dataset["trainA"], dataset["trainB"]
    test_horses, test_zebras = dataset["testA"], dataset["testB"]

    buffer_size = 256
    batch_size = 1

    # Apply the preprocessing operations to the training data
    train_horses = (
        train_horses.map(preprocess_train_image, num_parallel_calls=autotune)
            .cache()
            .shuffle(buffer_size)
            .batch(batch_size)
    )
    train_zebras = (
        train_zebras.map(preprocess_train_image, num_parallel_calls=autotune)
            .cache()
            .shuffle(buffer_size)
            .batch(batch_size)
    )

    # Apply the preprocessing operations to the test data
    test_horses = (
        test_horses.map(preprocess_test_image, num_parallel_calls=autotune)
            .cache()
            .shuffle(buffer_size)
            .batch(batch_size)
    )
    test_zebras = (
        test_zebras.map(preprocess_test_image, num_parallel_calls=autotune)
            .cache()
            .shuffle(buffer_size)
            .batch(batch_size)
    )

    # _, ax = plt.subplots(4, 2, figsize=(10, 15))
    # for i, samples in enumerate(zip(train_horses.take(4), train_zebras.take(4))):
    #     horse = (((samples[0][0] * 127.5) + 127.5).numpy()).astype(np.uint8)
    #     zebra = (((samples[1][0] * 127.5) + 127.5).numpy()).astype(np.uint8)
    #     ax[i, 0].imshow(horse)
    #     ax[i, 1].imshow(zebra)
    # plt.show()

    # Create cycle gan model
    cycle_gan_model = CycleGAN(input_shape=input_img_size)

    # Compile the model
    cycle_gan_model.compile(
        gen_G_optimizer=keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5),
        gen_F_optimizer=keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5),
        disc_X_optimizer=keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5),
        disc_Y_optimizer=keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5),
        gen_loss_fn=generator_loss_fn,
        disc_loss_fn=discriminator_loss_fn,
    )

    # cycle_gan_model.summary()

    # Define output shape for Model.save()
    # cycle_gan_model.compute_output_shape(input_shape=(None,)+input_img_size)
    # cycle_gan_model.save(r"C:\Bastien\Python\CycleGAN\output\cyclegan_test.hdf5")

    # Callbacks
    plotter = GANMonitor(test_dataset=test_horses)
    checkpoint_filepath = r"/export/home/users/brigaud/Output/CycleGAN/horse2zebra/cyclegan_checkpoints.{epoch:03d}"
    model_checkpoint_callback = keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath, save_weights_only=True)

    # Here we will train the model for just one epoch as each epoch takes around
    # 7 minutes on a single P100 backed machine.
    cycle_gan_model.fit(tf.data.Dataset.zip((train_horses, train_zebras)), epochs=10,
                        callbacks=[model_checkpoint_callback])


if __name__ == '__main__':
    main()
