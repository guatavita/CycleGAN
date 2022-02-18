# Created by Bastien Rigaud at 18/02/2022
# Bastien Rigaud, PhD
# Laboratoire Traitement du Signal et de l'Image (LTSI), INSERM U1099
# Campus de Beaulieu, Université de Rennes 1
# 35042 Rennes, FRANCE
# bastien.rigaud@univ-rennes1.fr
# Description:

import glob
import copy

from Base_Deeplearning_Code.Data_Generators.TFRecord_to_Dataset_Generator import DataGeneratorClass
from Base_Deeplearning_Code.Data_Generators.Image_Processors_Module.src.Processors.TFDataSetProcessors import \
    Random_Noise, Repeat_Channel, Cast_Data, Return_Outputs
# data augmentation and image processor
from Image_Processors_Utils.Image_Processor_Utils import *


def return_generator(base_path, image_keys=('image', 'annotation'), interp_keys=('bilinear', 'nearest'),
                          filling_keys=('constant', 'constant'), dtype_keys=('float16', 'float16'), is_validation=False,
                          model_desc='model_name', mean_val=0.0, std_val=1.0, nb_channel=1, image_size=(512, 512),
                          threshold_flag=None, threshold_value=255.0, debug=False, batch_size=1, max_noise=0.0,
                          scale_aug=1.0, crop_aug=False, lr_flip_aug=False, ud_flip_aug=False, rotation_angle_aug=0.0,
                          translation_aug=0.0, per_img_std=False, prep_tfrecord_cache=True, shuffle=True,
                          per_patient_std=False, contrast_aug=False, ct_clip=False):

    base_processors = [Expand_Dimensions_Per_Key(axis=-1, image_keys=image_keys),
                       Ensure_Image_Key_Proportions(image_rows=image_size[0], image_cols=image_size[1],
                                                    preserve_aspect_ratio=True,
                                                    image_keys=image_keys,
                                                    interp=interp_keys)]
    if ct_clip:
        base_processors += [Threshold_Images(image_keys=image_keys, lower_bounds=(-1000,), upper_bounds=(1500,), divides=(False,))]

    if not is_validation:
        tf_path = [os.path.join(base_path, 'train')]
    else:
        tf_path = [os.path.join(base_path, 'validation')]

    print(tf_path)
    generator = DataGeneratorClass(record_paths=tf_path, delete_old_cache=False, debug=debug)
    processors = []
    processors += base_processors
    augmentation = []

    if not is_validation:

        if contrast_aug != 0.0:
            augmentation += [Random_Contrast(image_keys=image_keys, lower_bounds=(1.0 - contrast_aug,),
                                             upper_bounds=(1.0 + contrast_aug,))]

        if lr_flip_aug:
            augmentation += [Random_Left_Right_flip(image_keys=image_keys)]

        if ud_flip_aug:
            augmentation += [Random_Up_Down_flip(image_keys=image_keys)]

        if rotation_angle_aug != 0.0:
            augmentation += [Random_Rotation(image_keys=image_keys, interp=interp_keys,
                                             angle=rotation_angle_aug, filling=filling_keys, dtypes=dtype_keys)]

        if translation_aug != 0.0:
            augmentation += [Random_Translation(image_keys=image_keys, interp=interp_keys,
                                                translation_x=translation_aug, translation_y=translation_aug,
                                                filling=filling_keys, dtypes=dtype_keys)]

        if scale_aug != 1.0 and scale_aug > 0.0:
            augmentation += [
                Random_Crop_and_Resize(min_scale=scale_aug, image_rows=image_size[0], image_cols=image_size[1],
                                       image_keys=image_keys, interp=interp_keys)]

    if crop_aug:
        augmentation += [Central_Crop_Img(image_keys=image_keys)]

    if mean_val != 0.0 and std_val != 1.0:
        print("{} {}".format(mean_val, std_val))
        augmentation += [Normalize_Images(keys=image_keys, mean_values=(mean_val,), std_values=(std_val,))]
        augmentation += [Threshold_Images(image_keys=image_keys, lower_bounds=(-3.55,), upper_bounds=(3.55,),
                                          divides=(False,))]
        # augmentation += [AddConstantToImages(keys=('image',), values=(3.55,))]
        # output should be Z-norm and min-max norm to [0,255]
        # augmentation += [MultiplyImagesByConstant(keys=('image',), values=(threshold_value / 7.10,))]

    if per_patient_std:
        augmentation += [Per_Patient_ZNorm(image_keys=image_keys, dtypes=dtype_keys)]

    if per_img_std:
        augmentation += [Per_Image_Z_Normalization(image_keys=image_keys)]

    if not is_validation:
        if max_noise > 0.0:
            augmentation += [Random_Noise(max_noise=max_noise)]

    if threshold_flag:
        augmentation += [Per_Image_MinMax_Normalization(image_keys=image_keys, threshold_value=threshold_value)]

    if nb_channel > 1:
        augmentation += [Repeat_Channel(axis=-1, repeats=nb_channel, on_images=True, on_annotations=False)]

    if not is_validation:
        processors += [Cast_Data(keys=image_keys, dtypes=dtype_keys)]
        if shuffle:
            processors += [{'shuffle': len(generator)}]
        processors += [{'cache': os.path.join(base_path, 'tf_cache', model_desc, 'train')}]
        if shuffle:
            processors += [{'shuffle': len(generator)}]
    else:
        processors += [Cast_Data(keys=image_keys, dtypes=dtype_keys),
                       {'cache': os.path.join(base_path, 'tf_cache', model_desc, 'validation')}]
        # if shuffle:
        #     processors += [{'shuffle': len(generator) // batch_size}]

    processors += augmentation
    processors += [
        Return_Outputs({'inputs': list(image_keys)}),
        {'batch': batch_size},
        {'repeat'},
    ]

    generator.compile_data_set(image_processors=processors, debug=debug)

    if prep_tfrecord_cache:
        if not is_validation:
            cache = glob.glob(os.path.join(base_path, 'tf_cache', model_desc, 'train', '*cache.tfrecord*'))
        else:
            cache = glob.glob(os.path.join(base_path, 'tf_cache', model_desc, 'validation', '*cache.tfrecord*'))
        if not cache:
            temp = iter(generator.data_set)
            for i in range(len(generator)):
                next(temp)

    return generator
