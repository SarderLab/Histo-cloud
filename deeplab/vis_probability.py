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

"""Segmentation results visualization on a given set of WSI.
See model.py for more details and usage.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
sys.path.append("..")

import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=FutureWarning)
    warnings.filterwarnings("ignore",category=RuntimeWarning)
    import os
    import time
    from large_image.cache_util import cachesClear
    import numpy as np
    from six.moves import range
    import tensorflow as tf
    from tensorflow.contrib import quantize as contrib_quantize
    from tensorflow.contrib import training as contrib_training
    from deeplab import common
    from deeplab import model
    from deeplab.datasets import wsi_data_generator
    from deeplab.utils import save_annotation
    from deeplab.utils.mask_to_xml import mask_to_xml
    from deeplab.utils.xml_to_json import convert_xml_json
    from scipy.special import softmax

flags = tf.app.flags

FLAGS = flags.FLAGS

flags.DEFINE_string('gpu', '0', 'GPU id used for training')

flags.DEFINE_string('savedir', '.', 'directory to save xmls')

flags.DEFINE_string('master', '', 'BNS name of the tensorflow server')

# Settings for log directories.

flags.DEFINE_string('vis_logdir', None, 'Where to write the event logs.')

flags.DEFINE_string('checkpoint_dir', None, 'Directory of model checkpoints.')

# Settings for visualizing the model.

flags.DEFINE_integer('vis_batch_size', 1,
                     'The number of images in each batch during evaluation.')

flags.DEFINE_integer('vis_crop_size', 2000,
                  'Crop size [size, size] for visualization.')

flags.DEFINE_integer('wsi_downsample', 2,
                  'Downsample rate of WSI used during training.')

flags.DEFINE_integer('tile_step', 500,
                  'Number of times the patch grid overlaps during testing.')

flags.DEFINE_integer('vis_remove_border', 100,
                   'Number of pixels on the patch boarder that are not used.')

flags.DEFINE_float('simplify_contours', 0.001,
                   'The amount of approximation used to simplify the predicted boundaries - we dont recommend setting this above a value of 0.01 (uses the Douglas-Peucker Algorythm epsilon).')

flags.DEFINE_integer('eval_interval_secs', 60 * 5,
                     'How often (in seconds) to run evaluation.')

# For `xception_65`, use atrous_rates = [12, 24, 36] if output_stride = 8, or
# rates = [6, 12, 18] if output_stride = 16. For `mobilenet_v2`, use None. Note
# one could use different atrous_rates/output_stride during training/evaluation.
flags.DEFINE_multi_integer('atrous_rates', None,
                           'Atrous rates for atrous spatial pyramid pooling.')

flags.DEFINE_integer('output_stride', 16,
                     'The ratio of input to output spatial resolution.')

# Change to [0.5, 0.75, 1.0, 1.25, 1.5, 1.75] for multi-scale test.
flags.DEFINE_multi_float('eval_scales', [1.0],
                         'The scales to resize images for evaluation.')

# Change to True for adding flipped images during test.
flags.DEFINE_bool('add_flipped_images', False,
                  'Add flipped images for evaluation or not.')

flags.DEFINE_integer(
    'quantize_delay_step', -1,
    'Steps to start quantized training. If < 0, will not quantize model.')

# Dataset settings.

flags.DEFINE_string('dataset', 'wsi_dataset',
                    'Name of the segmentation dataset.')

flags.DEFINE_integer('min_size', 0,
                  'Minimum size of the detected regions.')

flags.DEFINE_integer('num_classes', 2,
                  'Number of output classes.')

flags.DEFINE_string('class_names', None,
                  'Names of output classes for json conversion.')

flags.DEFINE_string('dataset_dir', None, 'Where the dataset reside.')

flags.DEFINE_boolean('save_json_annotation', False,
                     'Save the predictions in .json format for HistomicsTK.')

flags.DEFINE_string('json_filename', 'annotation.anot', '*.json annotation filename.')

# The folder where semantic segmentation predictions are saved.
_SEMANTIC_PREDICTION_SAVE_FOLDER = 'segmentation_results'

# The folder where raw semantic segmentation predictions are saved.
_RAW_SEMANTIC_PREDICTION_SAVE_FOLDER = 'raw_segmentation_results'

# The format to save image.
_IMAGE_FORMAT = '%06d_image'

# The format to save prediction
_PREDICTION_FORMAT = '%06d_prediction'


def _process_batch(sess, slide_mask, offset, semantic_predictions, semantic_probablities,
                   image_names, mask_size, border, downsample, extra_downsample, image_heights,
                   image_widths, image_id_offset,
                   raw_save_dir, train_id_to_eval_id=None):
  """Evaluates one single batch qualitatively.

  Args:
    sess: TensorFlow session.
    semantic_predictions: One batch of semantic segmentation predictions.
    semantic_probablities: One batch of semantic segmentation probabilities (softmax).
    image_names: Image names.
    mask_size: [y,x] dimentions of the mask
    image_heights: Image heights.
    image_widths: Image widths.
    image_id_offset: Image id offset for indexing images.
    raw_save_dir: The directory where the raw predictions will be saved.
    train_id_to_eval_id: A list mapping from train id to eval id.
  """
  # t = time.time()
  (semantic_predictions,
   semantic_probablities,
   image_names,
   image_heights,
   image_widths) = sess.run([semantic_predictions, semantic_probablities,
                             image_names, image_heights, image_widths])
  # print('\nNN time  : {}'.format(time.time()-t))
  # t = time.time()

  border = int(round(border/extra_downsample))
  num_image = semantic_predictions.shape[0]
  for i in range(num_image):
    image_height = np.squeeze(image_heights[i])/extra_downsample
    image_width = np.squeeze(image_widths[i])/extra_downsample
    semantic_prediction = np.squeeze(semantic_predictions[i])
    semantic_probability = semantic_probablities[i]
    image_filename = image_names[i].decode()

    # populate wsi mask
    Ystart = float(image_filename.split('-')[-2]) - offset['Y']
    Ystart /= downsample*extra_downsample
    Ystart = int(round(Ystart))+border

    Xstart = float(image_filename.split('-')[-3]) - offset['X']
    Xstart /= downsample*extra_downsample
    Xstart = int(round(Xstart))+border

    Ystop = min(int(round(Ystart+image_height-(border*2))), mask_size[0])
    Xstop = min(int(round(Xstart+image_width-(border*2))), mask_size[1])

    # print('\n')
    # print(mask_size)
    # print(border)
    # print(Xstart, Xstop, Ystart, Ystop)
    # print(np.shape(slide_mask[Ystart:Ystop, Xstart:Xstop]))
    # print(np.shape(semantic_prediction[border:Ystop-Ystart+border, border:Xstop-Xstart+border]))
    # print(np.shape(slide_mask))
    # semantic_prediction = np.ones(np.shape(semantic_prediction)) + semantic_prediction

    slide_mask[Ystart:Ystop, Xstart:Xstop,:] = slide_mask[Ystart:Ystop,
            Xstart:Xstop,:] + semantic_probability[border:Ystop-Ystart
            +border, border:Xstop-Xstart+border,:]

  # print('Mask time: {}'.format(time.time()-t))
  return slide_mask


def main(unused_argv):
  os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
  os.environ["CUDA_VISIBLE_DEVICES"]=FLAGS.gpu

  tf.logging.set_verbosity(tf.logging.INFO)

  # Get dataset-dependent information.
  dataset = wsi_data_generator.Dataset(
      dataset_name=FLAGS.dataset,
      dataset_dir=FLAGS.dataset_dir,
      num_of_classes=FLAGS.num_classes,
      downsample=FLAGS.wsi_downsample,
      tile_step=FLAGS.tile_step,
      batch_size=FLAGS.vis_batch_size,
      crop_size=FLAGS.vis_crop_size,
      min_resize_value=FLAGS.min_resize_value,
      max_resize_value=FLAGS.max_resize_value,
      resize_factor=FLAGS.resize_factor,
      model_variant=FLAGS.model_variant,
      is_training=False,
      should_shuffle=False,
      should_repeat=False)

  assert FLAGS.vis_remove_border*2 < FLAGS.vis_crop_size and FLAGS.vis_remove_border*2 < FLAGS.vis_crop_size - FLAGS.tile_step

  if os.path.isfile(FLAGS.dataset_dir):
      slides = [FLAGS.dataset_dir]
  else:
      # get all WSI in test set
      slides = dataset._get_all_files(with_xml=False, save_mask=False)

  broken_slides = []

  with tf.Graph().as_default():
      checkpoint_path = FLAGS.checkpoint_dir
      config = tf.ConfigProto()
      config.gpu_options.allow_growth = True
      scaffold = tf.train.Scaffold(init_op=tf.global_variables_initializer())
      session_creator = tf.train.ChiefSessionCreator(
            scaffold=scaffold,
            master=FLAGS.master,
            config=config,
            checkpoint_filename_with_path=checkpoint_path)

      for slide in slides:
          print('Working on: [{}]'.format(slide))

          # try:
          train_id_to_eval_id = None
          raw_save_dir = None
          iterator, num_samples, tissue_offset, tissue_size = dataset.get_one_shot_iterator_grid(slide)

          # except Exception as e:
          #     print(e)
          #     print('!!! Faulty slide: skipping [{}] !!!'.format(slide))
          #     broken_slides.append(slide)
          #     continue

          samples = iterator.get_next()

          model_options = common.ModelOptions(
              outputs_to_num_classes={common.OUTPUT_TYPE: dataset.num_of_classes},
              crop_size=[FLAGS.vis_crop_size,FLAGS.vis_crop_size],
              atrous_rates=FLAGS.atrous_rates,
              output_stride=FLAGS.output_stride)

          tf.logging.info('Performing WSI patch detection.\n')
          predictions = model.predict_labels(
                samples[common.IMAGE],
                model_options=model_options,
                image_pyramid=FLAGS.image_pyramid)

          probabilities = predictions[common.OUTPUT_TYPE+'_prob']
          predictions = predictions[common.OUTPUT_TYPE]

          # do not upsample predictions in network output
          if not model_options.prediction_with_upsampled_logits:
              extra_downsample = int(FLAGS.vis_crop_size / int(predictions.shape[1]))
          else: extra_downsample=1

          # get tissue region size and create empty wsi mask
          def get_downsampled_size(size, downsample=FLAGS.wsi_downsample*extra_downsample):
              size /= downsample
              return int(np.ceil(size))
          mask_size = [get_downsampled_size(tissue_size[0]), get_downsampled_size(tissue_size[1])]
          os.system("printf 'Creating a slide mask {} pixels\n'".format(mask_size))
          slide_mask = np.zeros([mask_size[0], mask_size[1], FLAGS.num_classes], dtype=np.float16)

          tf.train.get_or_create_global_step()
          if FLAGS.quantize_delay_step >= 0:
              contrib_quantize.create_eval_graph()

          with tf.train.MonitoredSession(
              session_creator=session_creator, hooks=None) as sess:
              batch = 0
              image_id_offset = 0
              print('\n')
              while not sess.should_stop():
                  # tf.logging.info('Visualizing batch %d', batch + 1)
                  os.system("printf 'Working on [{}] patch: [{} of {}]\n'".format(os.path.basename(slide), min(batch, num_samples), num_samples))
                  slide_mask = _process_batch(sess=sess,
                                 slide_mask=slide_mask,
                                 offset=tissue_offset,
                                 semantic_predictions=predictions,
                                 semantic_probablities=probabilities,
                                 image_names=samples[common.IMAGE_NAME],
                                 mask_size=mask_size,
                                 border=FLAGS.vis_remove_border,
                                 downsample=FLAGS.wsi_downsample,
                                 extra_downsample=extra_downsample,
                                 image_heights=samples[common.HEIGHT],
                                 image_widths=samples[common.WIDTH],
                                 image_id_offset=image_id_offset,
                                 raw_save_dir=raw_save_dir,
                                 train_id_to_eval_id=train_id_to_eval_id)
                  image_id_offset += FLAGS.vis_batch_size
                  batch += FLAGS.vis_batch_size

          # clear large image caches
          cachesClear()

          '''
          # make region heatmap JSON
          stride = 4

          # tissue mask
          from PIL import Image
          ds = FLAGS.wsi_downsample*extra_downsample*stride

          # tissue_mask = Image.fromarray(tissue_mask)
          # tissue_mask_size = np.array([float(tissue_mask.size[0]), float(tissue_mask.size[1])])
          # tissue_mask_final_size = tissue_mask_size / (mask_scale * ds) - [tissue_offset['X']/ds,tissue_offset['Y']/ds]
          # tissue_mask_final_size = [np.int(np.round(tissue_mask_final_size[0])),np.int(np.round(tissue_mask_final_size[1]))]
          # tissue_mask = tissue_mask.resize(tissue_mask_final_size)
          # tissue_mask = np.array(tissue_mask)>128
          # tissue_mask = tissue_mask[int(round(tissue_offset['X']/ds)):, int(round(tissue_offset['Y']/ds)):]
          # strided_mask = slide_mask[::stride,::stride,:]
          # mask_size = strided_mask.shape
          # # get tissue mask the right size
          # tissue_mask = tissue_mask[:mask_size[0],:mask_size[1]]
          # pad_size = (max(mask_size[0] - tissue_mask.shape[0], 0), max(mask_size[1] - tissue_mask.shape[1], 0))
          # tissue_mask = np.pad(tissue_mask,[[0,pad_size[0]],[0,pad_size[1]]])

          cutoff = 1/(FLAGS.num_classes*3) # cutoff low values
          heatmap = {"type":"heatmap", "radius":FLAGS.wsi_downsample*extra_downsample, "colorRange": ["rgba(255,255,0,0)", "rgba(255,255,0,.3)", "rgba(255,190,0,.4)", "rgba(255,0,0,.5)"], "rangeValues": [cutoff, cutoff*1.5, 1.5/FLAGS.num_classes, 1], "normalizeRange":True}
          mask_sum = strided_mask.sum(2)
          mask_sum_mask = mask_sum==0
          np.place(strided_mask[:,:,0],mask_sum_mask,1)
          np.place(mask_sum,mask_sum_mask,1)
          strided_mask /=np.repeat(np.expand_dims(mask_sum,-1), FLAGS.num_classes,-1)
          single_mask = strided_mask[:,:,1]

          # np.place(single_mask, tissue_mask==0, 0) # to remove non tissue
          values = single_mask.flatten()
          values_mask = values>(cutoff)
          values = np.round(values[values_mask],3)
          np.place(values, values>1, 1)
          y_idx, x_idx = np.indices(single_mask.shape)
          y_idx = y_idx.flatten()[values_mask]
          x_idx = x_idx.flatten()[values_mask]
          # y_idx = y_idx.flatten()[values_mask]
          # x_idx = x_idx.flatten()[values_mask]
          z_idx = np.zeros_like(x_idx)
          points = np.empty((x_idx.size*4), dtype = np.float32)
          points[0::4] = x_idx*ds + tissue_offset['X']
          points[1::4] = y_idx*ds + tissue_offset['Y']
          points[2::4] = z_idx
          points[3::4] = values
          heatmap["points"] = points.reshape([-1,4]).tolist()

          data = {"name":"IFTA-logits","elements":[heatmap]}
          import json
          dirpath = os.path.dirname(slide)
          basename = os.path.splitext(os.path.basename(slide))[0]
          anot_filename = '{}/{}-IFTA-logits.anot'.format(dirpath, basename)
          print('dumping JSON')
          with open(anot_filename, 'w') as annotation_file:
              json.dump(data, annotation_file, sort_keys=False)

          continue
          # '''

          thresholds = np.linspace(0.99,1,101)
          num_threshs = len(thresholds)
          for iter, threshold in enumerate(thresholds):

              print('\t{} of {} | working on threshold: {}'.format(iter+1, num_threshs, round(threshold,4)), end="\r", flush=True)

              # weight classes
              # weights = [1-threshold,1-threshold,threshold,1-threshold]
              weights = [1-threshold,threshold]
              weighted_mask = np.argmax(slide_mask*weights, -1).astype(np.uint8)

              if FLAGS.save_json_annotation:
                  anot_filename = FLAGS.json_filename
                  print('\ncreating annotation file: [{}]'.format(anot_filename))
                  root = mask_to_xml(xml_path=anot_filename, mask=slide_mask, downsample=FLAGS.wsi_downsample*extra_downsample, min_size_thresh=FLAGS.min_size, simplify_contours=FLAGS.simplify_contours, return_root=True, maxClass=FLAGS.num_classes-1, offset=tissue_offset)
                  compartments = FLAGS.class_names.split(',')
                  json_data = convert_xml_json(root, compartments)
                  import json
                  with open(anot_filename, 'w') as annotation_file:
                      json.dump(json_data, annotation_file, indent=2, sort_keys=False)
                  del json_data, root, anot_filename

              else:
                  dirpath = os.path.dirname(slide)
                  basename = os.path.splitext(os.path.basename(slide))[0]
                  annot_dir = '{}/arteriole-{}/'.format(dirpath,FLAGS.savedir)
                  if not os.path.exists(annot_dir):
                      os.mkdir(annot_dir)
                  anot_filename = '{}{}-threshold={}.xml'.format(annot_dir,basename, str(round(threshold,4)).replace('.','_'))
                  mask_to_xml(xml_path=anot_filename, mask=weighted_mask, downsample=FLAGS.wsi_downsample*extra_downsample, min_size_thresh=FLAGS.min_size, simplify_contours=FLAGS.simplify_contours, offset=tissue_offset)

          '''
          for iter, threshold in enumerate(thresholds):

              print('\t{} of {} | working on threshold: {}'.format(iter+1, num_threshs, round(threshold,4)), end="\r", flush=True)

              # weight classes
              weights = [1-threshold,1-threshold,1-threshold,threshold]
              # weights = [1-threshold,threshold]
              weighted_mask = np.argmax(slide_mask*weights, -1).astype(np.uint8)

              if FLAGS.save_json_annotation:
                  anot_filename = FLAGS.json_filename
                  print('\ncreating annotation file: [{}]'.format(anot_filename))
                  root = mask_to_xml(xml_path=anot_filename, mask=slide_mask, downsample=FLAGS.wsi_downsample*extra_downsample, min_size_thresh=FLAGS.min_size, simplify_contours=FLAGS.simplify_contours, return_root=True, maxClass=FLAGS.num_classes-1, offset=tissue_offset)
                  compartments = FLAGS.class_names.split(',')
                  json_data = convert_xml_json(root, compartments)
                  import json
                  with open(anot_filename, 'w') as annotation_file:
                      json.dump(json_data, annotation_file, indent=2, sort_keys=False)
                  del json_data, root, anot_filename

              else:
                  dirpath = os.path.dirname(slide)
                  basename = os.path.splitext(os.path.basename(slide))[0]
                  annot_dir = '{}/artery-{}/'.format(dirpath,FLAGS.savedir)
                  if not os.path.exists(annot_dir):
                      os.mkdir(annot_dir)
                  anot_filename = '{}{}-threshold={}.xml'.format(annot_dir,basename, str(round(threshold,4)).replace('.','_'))
                  mask_to_xml(xml_path=anot_filename, mask=weighted_mask, downsample=FLAGS.wsi_downsample*extra_downsample, min_size_thresh=FLAGS.min_size, simplify_contours=FLAGS.simplify_contours, offset=tissue_offset)
          '''

          del anot_filename
          del slide_mask, predictions

  if len(broken_slides) > 0:
      print('\n!!! The following slides are broken and were not run:')
      for slide in broken_slides:
          print('-\t[{}]'.format(slide))


if __name__ == '__main__':
  flags.mark_flag_as_required('checkpoint_dir')
  flags.mark_flag_as_required('dataset_dir')
  tf.app.run()
  print('\n\nall done.')
