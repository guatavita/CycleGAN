# Created by Bastien Rigaud at 18/02/2022
# Bastien Rigaud, PhD
# Laboratoire Traitement du Signal et de l'Image (LTSI), INSERM U1099
# Campus de Beaulieu, Université de Rennes 1
# 35042 Rennes, FRANCE
# bastien.rigaud@univ-rennes1.fr
# Description:

import os
import tensorflow as tf
from tensorflow.keras.callbacks import Callback
import numpy as np
import io
import itertools
import matplotlib
import matplotlib.pyplot as plt

from PlotScrollNumpyArrays.Plot_Scroll_Images import plot_scroll_Image

def norm_tensor(tensor, scale=tf.constant(255.0, dtype="float16")):
    numer = tensor - tf.reduce_min(tensor)
    denom = (tf.reduce_max(tensor) - tf.reduce_min(tensor)) + tf.constant(1e-6, dtype="float16")
    normed_array = numer / denom
    normed_array = tf.cast((normed_array * scale), dtype="uint8")
    return normed_array

def resize_figure(figure, image_rows, image_cols):
    # Add the batch dimension
    image = tf.expand_dims(figure, 0)
    image = tf.image.resize(image, (image_rows, image_cols), method='bilinear',
                            preserve_aspect_ratio=True)
    image = tf.image.resize_with_crop_or_pad(image, target_height=image_rows, target_width=image_cols)
    image = tf.cast(image, "uint8")
    return image

class Add_Cycle_Images(Callback):
    def __init__(self, log_dir, validation_data=None, number_of_images=5, image_rows=512, image_cols=512, frequency=5):
        super(Add_Cycle_Images, self).__init__()
        if validation_data is None:
            AssertionError('Need to provide validation data')
        self.validation_data = iter(validation_data)
        self.number_of_images = number_of_images
        self.image_rows = image_rows
        self.image_cols = image_cols
        self.frequency = frequency
        self.file_writer = tf.summary.create_file_writer(os.path.join(log_dir, 'val_cycle_images'))

    def plot_to_image(self, figure):
        """Converts the matplotlib plot specified by 'figure' to a PNG image and
        returns it. The supplied figure is closed and inaccessible after this call."""
        # Save the plot to a PNG in memory.
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        # Closing the figure prevents it from being displayed
        plt.close(figure)
        buf.seek(0)
        # Convert PNG buffer to TF image
        image = tf.image.decode_png(buf.getvalue(), channels=4)
        # Add the batch dimension
        image = tf.expand_dims(image, 0)
        return image

    def log_images(self):
        batch_id = 0
        for i in range(self.number_of_images):
            print("Reporting cycle images {}".format(i))
            real_x, real_y = next(self.validation_data)
            fake_y = self.model.gen_G(real_x)[batch_id]
            fake_x = self.model.gen_F(real_y)[batch_id]
            real_x, real_y = real_x[0][batch_id], real_y[0][batch_id]

            normed_x = norm_tensor(real_x)
            normed_y = norm_tensor(real_y)
            normed_fy = norm_tensor(tf.cast(fake_y, dtype=real_x.dtype))
            normed_fx = norm_tensor(tf.cast(fake_x, dtype=real_y.dtype))

            # order is important here
            figure_xfy = tf.concat([normed_x, normed_fy], axis=0)
            image_xfy = resize_figure(figure_xfy, image_rows=2*self.image_rows, image_cols=self.image_cols)

            figure_yfx = tf.concat([normed_y, normed_fx], axis=0)
            image_yfx = resize_figure(figure_yfx, image_rows=2*self.image_rows, image_cols=self.image_cols)
            if i == 0:
                out_xfy_image = image_xfy
                out_yfx_image = image_yfx
            else:
                out_xfy_image = tf.concat([out_xfy_image, image_xfy], axis=2)
                out_yfx_image = tf.concat([out_yfx_image, image_yfx], axis=2)

        return out_xfy_image, out_yfx_image

    def on_epoch_end(self, epoch, logs=None):
        # Log the confusion matrix as an image summary.
        if self.frequency != 0 and epoch != 0 and epoch % self.frequency == 0:
            with self.file_writer.as_default():
                out_xfy_image, out_yfx_image = self.log_images()
                tf.summary.image("Cycle Images X to fake Y", out_xfy_image, step=epoch)
                tf.summary.image("Cycle Images Y to fake X", out_yfx_image, step=epoch)
