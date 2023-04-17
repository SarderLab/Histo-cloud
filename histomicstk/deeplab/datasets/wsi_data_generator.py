# Lint as: python2, python3
# Copyright 2018 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Wrapper for providing semantic segmentaion data.

The SegmentationDataset class provides both images and annotations (semantic
segmentation and/or instance segmentation) for TensorFlow using WSIs and
Aperio XML annotations.

References:
  M. Everingham, S. M. A. Eslami, L. V. Gool, C. K. I. Williams, J. Winn,
  and A. Zisserman, The pascal visual object classes challenge a retrospective.
  IJCV, 2014.

  M. Cordts, M. Omran, S. Ramos, T. Rehfeld, M. Enzweiler, R. Benenson,
  U. Franke, S. Roth, and B. Schiele, "The cityscapes dataset for semantic urban
  scene understanding," In Proc. of CVPR, 2016.

  B. Zhou, H. Zhao, X. Puig, S. Fidler, A. Barriuso, A. Torralba, "Scene Parsing
  through ADE20K dataset", In Proc. of CVPR, 2017.
"""

import collections
import os
import tensorflow as tf
from deeplab import common
from deeplab import input_preprocess
from glob import glob
try:
    from utils.wsi_dataset_util_large_image import get_wsi_patch, get_patch_from_points, get_num_classes, get_grid_list, save_wsi_thumbnail_mask
except:
    from deeplab.utils.wsi_dataset_util_large_image import get_wsi_patch, get_patch_from_points, get_num_classes, get_grid_list, save_wsi_thumbnail_mask
try:
    from utils.xml_to_mask import write_minmax_to_xml
except:
    from deeplab.utils.xml_to_mask import write_minmax_to_xml

class Dataset(object):
  """Represents input dataset for deeplab model."""

  def __init__(self,
               dataset_name,
               dataset_dir,
               batch_size,
               crop_size,
               downsample,
               tile_step=None,
               include_background_prob = 0,
               augment_prob = 0,
               num_of_classes = None,
               wsi_ext=['.svs', '.ndpi', '.scn', '.czi'],
               min_resize_value=None,
               max_resize_value=None,
               resize_factor=None,
               min_scale_factor=1.,
               max_scale_factor=1.,
               scale_factor_step_size=0,
               model_variant=None,
               num_readers=1,
               is_training=False,
               should_shuffle=False,
               should_repeat=False,
               ignore_label=255):
    """Initializes the dataset.

    Args:
      dataset_name: Dataset name.
      dataset_dir: The directory of the dataset sources.
      batch_size: Batch size.
      crop_size: The size used to crop the image and label.
      downsample: The downsample of WSI patches used
      wsi_ext: A list of the possible file extensions of wsi files
      min_resize_value: Desired size of the smaller image side.
      max_resize_value: Maximum allowed size of the larger image side.
      resize_factor: Resized dimensions are multiple of factor plus one.
      min_scale_factor: Minimum scale factor value.
      max_scale_factor: Maximum scale factor value.
      scale_factor_step_size: The step size from min scale factor to max scale
        factor. The input is randomly scaled based on the value of
        (min_scale_factor, max_scale_factor, scale_factor_step_size).
      model_variant: Model variant (string) for choosing how to mean-subtract
        the images. See feature_extractor.network_map for supported model
        variants.
      num_readers: Number of readers for data provider.
      is_training: Boolean, if dataset is for training or not.
      should_shuffle: Boolean, if should shuffle the input data.
      should_repeat: Boolean, if should repeat the input data.

    """

    if model_variant is None:
      tf.logging.warning('Please specify a model_variant. See '
                         'feature_extractor.network_map for supported model '
                         'variants.')

    self.dataset_name = dataset_name
    self.dataset_dir = dataset_dir
    self.wsi_ext = wsi_ext
    self.batch_size = batch_size
    self.crop_size = crop_size
    self.downsample = downsample
    self.tile_step = tile_step
    self.augment_prob = augment_prob
    self.include_background_prob = include_background_prob
    self.min_resize_value = min_resize_value
    self.max_resize_value = max_resize_value
    self.resize_factor = resize_factor
    self.min_scale_factor = min_scale_factor
    self.max_scale_factor = max_scale_factor
    self.scale_factor_step_size = scale_factor_step_size
    self.model_variant = model_variant
    self.num_readers = num_readers
    self.is_training = is_training
    self.should_shuffle = should_shuffle
    self.should_repeat = should_repeat
    self.ignore_label = ignore_label

    if num_of_classes == None:
        self.num_of_classes = self._get_num_classes()
    else:
        self.num_of_classes = num_of_classes

  def _parse_function(self, image, label, imageID):
    """Function to parse the example proto.

    Args:
      [image, label].

    Returns:
      A dictionary with parsed image, and label.

    Raises:
      ValueError: Label is of wrong shape.
    """

    image.set_shape([self.crop_size, self.crop_size, 3])
    sample = {
        common.IMAGE: image,
    }

    if label is not None:
        label.set_shape([self.crop_size, self.crop_size, 1])
        sample[common.LABEL] = label

    sample[common.IMAGE_NAME] = imageID
    sample[common.HEIGHT] = self.crop_size
    sample[common.WIDTH] = self.crop_size

    return sample

  def get_one_shot_iterator(self):
    """Gets an iterator that iterates across the dataset once.
        Pulls random patches from WSI at runtime

    Returns:
      An iterator of type tf.data.Iterator.
    """

    #######################################################################

    wsi_paths = self._get_all_files()

    # setup tf dataset using py_function
    path_ds = tf.data.Dataset.from_tensor_slices(wsi_paths)

    if self.should_shuffle:
        path_ds = path_ds.shuffle(buffer_size=100)

    if self.should_repeat:
        path_ds = path_ds.repeat()  # Repeat forever for training.
    else:
        path_ds = path_ds.repeat(1)

    wsi_dataset = path_ds.map(lambda filename: tf.py_function(
            get_wsi_patch, [filename, self.crop_size, self.downsample, self.include_background_prob,
            self.augment_prob,self.ignore_label], [tf.float32,tf.uint8,tf.string]),
            num_parallel_calls=self.num_readers)


    wsi_dataset = wsi_dataset.map(self._parse_function, num_parallel_calls=self.num_readers)
    # wsi_dataset = wsi_dataset.map(self._preprocess_image, num_parallel_calls=self.num_readers)

    wsi_dataset = wsi_dataset.batch(batch_size=self.batch_size, drop_remainder=True)
    wsi_dataset = wsi_dataset.prefetch(buffer_size=2) # <-- very important for efficency
    return wsi_dataset.make_one_shot_iterator()

  def get_one_shot_iterator_grid(self, wsi_path):
    """Gets an iterator that iterates across the dataset once.
        Pulls grid of patches from WSI at runtime
        Used for prediction

    Returns:
      An iterator of type tf.data.Iterator.
    """

    #######################################################################

    # wsi_path = '{}/{}'.format(self.dataset_dir, self.slide_name)

    # open slide once globally for efficency
    import large_image
    wsi = large_image.getTileSource(wsi_path)

    # get grid of start points of patches
    points, length, tissue_offset, tissue_size = get_grid_list(wsi_path, self.crop_size, self.downsample, self.tile_step, wsi)

    # setup tf dataset using py_function
    points_ds = tf.data.Dataset.from_tensor_slices(points)

    wsi_dataset = points_ds.map(lambda point: tf.py_function(
            get_patch_from_points, [wsi_path, point, self.crop_size,
            self.downsample], [tf.float32,tf.uint8,tf.string]),
            num_parallel_calls=self.num_readers)

    wsi_dataset = wsi_dataset.map(self._parse_function, num_parallel_calls=self.num_readers)
    # wsi_dataset = wsi_dataset.map(self._preprocess_image, num_parallel_calls=self.num_readers)

    wsi_dataset = wsi_dataset.batch(batch_size=self.batch_size, drop_remainder=False)
    wsi_dataset = wsi_dataset.prefetch(buffer_size=1) # <-- very important for efficency
    return wsi_dataset.make_one_shot_iterator(), length, tissue_offset, tissue_size

  def _get_all_files(self, with_xml=True, save_mask=True):
    """Gets all the files to read data from.

    with_xml: find only slides with xml, else find only slides with no xml

    Returns:
      A list of input WSI files.
    """

    wsi_paths = []

    print('\n----------------------------------')
    print('searching for slides in:\n[{}]'.format(self.dataset_dir))

    for ext in self.wsi_ext:
        slides = list(glob('{}/*{}'.format(self.dataset_dir, ext)))
        for slide in slides:
            if with_xml:
                # check for annotaiton file
                xml_filename = '{}.xml'.format(slide.split(ext)[0])
                if os.path.isfile(xml_filename):
                    wsi_paths.append(slide)
                    # write to xml file to avoid parallel writes durring training
                    write_minmax_to_xml(xml_filename)
            if not with_xml:
                # check for missing annotaiton file
                if not os.path.isfile('{}.xml'.format(slide.split(ext)[0])):
                    wsi_paths.append(slide)

    wsi_paths = [str(path) for path in wsi_paths]

    wsi_paths_temp = []
    for wsi_path in wsi_paths:
        try:
            if save_mask:
                save_wsi_thumbnail_mask(wsi_path)
            wsi_paths_temp.append(wsi_path)
        except:
            print('skipping: [{}], multi-resolution slide broken'.format(wsi_path))
    wsi_paths = wsi_paths_temp

    image_count = len(wsi_paths)
    print('\nfound: [{}] slides from WSI dataset'.format(image_count))
    print('----------------------------------\n')

    assert image_count > 0, 'No data found in: [{}]'.format(self.dataset_dir)

    return wsi_paths

  def _get_num_classes(self):
    """Gets the number of classes in the dataset from XML annotaitons.

    Returns:
      number of classes (int).
    """
    classes = 0
    slides = self._get_all_files()
    for slide in slides:
        c = get_num_classes('{}.xml'.format(os.path.splitext(slide)[0]), self.ignore_label)
        classes = max(classes, c)
    print('\n-----------------------')
    print('Found [{}] data classes'.format(classes))
    print('-----------------------\n')
    return classes
