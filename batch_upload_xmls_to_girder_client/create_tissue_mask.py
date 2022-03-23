import os, cv2, random, time
import numpy as np
import scipy
import scipy.misc
import openslide
import pandas as pd
import lxml.etree as ET
from PIL import ImageFilter, Image
from mask_to_xml import mask_to_xml
from glob import glob

### args ###

savedir = 'tissue_masks'

############

if not os.path.exists(savedir):
    os.mkdir(savedir)

def save_wsi_thumbnail_mask(filename):
    '''
    saves or returns a low resolution png mask of the tissue location in a WSI

    '''

    try: filename = filename.numpy()
    except: filename = filename
    wsi = openslide.OpenSlide(filename)
    thumbnail_size = 2000

    def find_tissue_mask():
        thumbnail = wsi.get_thumbnail((thumbnail_size,thumbnail_size))
        thumbnail_blurred = np.array(thumbnail.filter(ImageFilter.GaussianBlur(radius=10)))
        ret2,mask = cv2.threshold(thumbnail_blurred[:,:,0],0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        kernel = np.ones((5,5),np.uint8)
        mask = cv2.erode(mask,kernel,iterations = 1)
        mask[mask==0] = 1
        mask[mask==255] = 0
        return mask

    slide_mask = find_tissue_mask()

    l_dims = wsi.level_dimensions
    thumbnail_size = min(2000., max(l_dims[0]))

    level_dims = l_dims[0]
    thumbnail_size = float(max(slide_mask.shape))
    mask_scale = thumbnail_size/max(level_dims)

    return slide_mask, mask_scale

xml_color=[65280, 65535, 33023, 255, 16711680]

slides = glob('*.svs')
broken = []
for slide in slides:
    try:
        mask, mask_scale = save_wsi_thumbnail_mask(slide)

        xml_path = '{}/{}.xml'.format(savedir, os.path.splitext(slide)[0])
        if not os.path.exists(xml_path):
            mask_to_xml(xml_path, mask, downsample=1/mask_scale, xml_color=xml_color)
            print('saved: [{}]'.format(xml_path))
    except: broken.append(slide)

print(brokes)
