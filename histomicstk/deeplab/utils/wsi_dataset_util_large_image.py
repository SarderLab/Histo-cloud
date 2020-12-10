import os, cv2, random, time
import numpy as np
import scipy
import scipy.misc
import large_image
import openslide
import pandas as pd
import lxml.etree as ET
from PIL import ImageFilter, Image
import matplotlib.pyplot as plt
from skimage.transform import PiecewiseAffineTransform, warp
from skimage.color import rgb2hsv,hsv2rgb,rgb2lab,lab2rgb
from skimage import exposure
try:
    from utils.xml_to_mask import xml_to_mask, get_num_classes, write_minmax_to_xml
except:
    from deeplab.utils.xml_to_mask import xml_to_mask, get_num_classes, write_minmax_to_xml

################################################################################
################################ main function #################################
################################################################################

def get_wsi_patch(filename, patch_size=256, downsample=[1], include_background_prob=0.1, augment=0):
    '''
    takes a wsi and returns a random patch of patch_size
    downsample = >1 downsample of patch (passed as a list)
    augment = [0,1] the percent of data that will be augmented

    '''
    try: augment = augment.numpy()
    except: pass
    try: include_background_prob = include_background_prob.numpy()
    except: pass
    try: patch_size = patch_size.numpy()
    except: pass
    try: filename = filename.numpy()
    except: pass
    try: downsample = downsample.numpy()
    except: pass

    # choose random downsample
    downsample = random.choice(downsample)

    wsi = large_image.getTileSource(filename)

    l_dims = get_slide_size(wsi=wsi)
    level = wsi.get_best_level_for_downsample(downsample + 0.1)

    try:
        base_name = filename.decode().split('.')[0]
    except:
        base_name = filename.split('.')[0]
    xml_path = '{}.xml'.format(base_name)

    slide_mask = get_slide_mask(filename)

    # test for xml and choose random annotated class for patch
    if os.path.isfile(xml_path):
        class_num = get_num_classes(xml_path)
        class_num = int(round(np.random.uniform(low=0, high=class_num-1)))
    else:
        class_num = 0

    region, mask, x_start, y_start = get_patch(wsi, xml_path, class_num, l_dims, level, slide_mask, patch_size, filename, downsample, include_background_prob, augment)

    # scale to [-1,1]
    # region = scale_patch(region)

    # region = get_random_patch(wsi, l_dims, level, mask, patch_size, filename, downsample, include_background_prob, augment)

    # region = np.transpose(region, (2,0,1)) # [CWH]
    imageID = '{}-{}-{}-{}'.format(base_name.split('/')[-1], x_start, y_start, downsample)

    return [region, mask, imageID]

def get_patch_from_points(filename, point, patch_size, downsample=1):
    '''
    takes a wsi filename and tuple (x,y) location and returns a patch of patch_size
    downsample = >1 downsample of patch

    '''
    try: patch_size = patch_size.numpy()
    except: pass
    try: downsample = downsample.numpy()
    except: pass
    try: point = point.numpy()
    except: pass
    try: filename = filename.numpy()
    except: pass
    try: filename = filename.decode()
    except: pass
    base_name = filename.split('.')[0]

    # t = time.time()
    wsi = large_image.getTileSource(filename)
    # print('t0: {}'.format(time.time()-t))

    # set native mag if None
    nativeMag = wsi.getNativeMagnification()['magnification']
    if nativeMag is None:
        mm_x = wsi.getNativeMagnification()['mm_x']
        if mm_x is None:
            wsi.getNativeMagnification = lambda: {"magnification": 40, "mm_x": 0.0005, "mm_y": 0.0005}
        else:
             wsi.getNativeMagnification = lambda: {"magnification": 0.01/mm_x, "mm_x": mm_x, "mm_y": wsi.getNativeMagnification()['mm_y']}
        nativeMag = wsi.getNativeMagnification()['magnification']

    mag = nativeMag/float(downsample)
    scaled_patch_size = (patch_size*downsample)+1 # add 1 to avoid rounding errors

    region, _ = wsi.getRegion(region=dict(left=point[0],top=point[1], width=scaled_patch_size, height=scaled_patch_size, units='base_pixels'), scale=dict(magnification=mag), format=large_image.tilesource.TILE_FORMAT_NUMPY)
    # print('t1: {}'.format(time.time()-t))

    # remove any excess
    region = region[:patch_size,:patch_size,:3]
    assert region.shape == (patch_size,patch_size,3)

    # scale to [-1,1]
    # region = scale_patch(region)

    # region = np.transpose(region, (2,0,1)) # [CWH]
    imageID = '{}-{}-{}-{}'.format(base_name.split('/')[-1], point[0], point[1], downsample)

    # create zeros mask to pass - NOT USED LATER
    mask = np.zeros([patch_size,patch_size], dtype=np.uint8)
    # print('t2: {}'.format(time.time()-t))

    return [region, mask, imageID]

################################################################################
################################# subfunctions #################################
################################################################################

def get_patch(wsi, xml_path, annotationID, l_dims, level, slide_mask, patch_size, filename, downsample, include_background_prob, augment):

    og_patch_size = patch_size
    if augment > 0:
        # pad for affine
        patch_size = patch_size+4

    while True:

        if level == -1: # if no resolution works return white region
            print('{} broken | using white patch...'.format(filename))
            return np.ones((og_patch_size, og_patch_size,3), dtype=np.uint8)*255, np.zeros((og_patch_size, og_patch_size), dtype=np.uint8), 0, 0

        try:

            level_dims = l_dims
            level_downsample = wsi.level_downsamples[level]
            thumbnail_size = float(max(slide_mask.shape))
            mask_scale = thumbnail_size/max(level_dims)
            scale_factor = int(round(downsample / level_downsample))
            patch_width = patch_size*scale_factor

            if annotationID != 0:
                # parse xml and get root
                tree = ET.parse(xml_path)
                root = tree.getroot()

                write_minmax_to_xml(xml_path, tree)

                locations = []
                # find all regions in annotation
                Verts = root.findall("./Annotation[@Id='{}']/*/*/Vertices".format(annotationID))

                # check if Annotation has any regions if not pick random location
                if len(Verts) > 0:
                    # get minmax bounds from annotations
                    for Vert in Verts:
                        # get minmax points
                        Xmin = np.int32(Vert.attrib['Xmin'])
                        Ymin = np.int32(Vert.attrib['Ymin'])
                        Xmax = np.int32(Vert.attrib['Xmax'])
                        Ymax = np.int32(Vert.attrib['Ymax'])
                        locations.append([Xmin,Xmax,Ymin,Ymax])

                    location = random.choice(locations)
                    # find point in a random annotation
                    # add noise to the center point
                    half_patch = (patch_width*level_downsample/2)
                    x_noise = np.random.uniform(low=-half_patch, high=half_patch)
                    y_noise = np.random.uniform(low=-half_patch, high=half_patch)
                    # select random point in region
                    x_start = int(round(np.random.uniform(low=location[0], high=location[1]) - half_patch + x_noise))
                    y_start = int(round(np.random.uniform(low=location[2], high=location[3]) - half_patch + y_noise))

                else: # if there are no annotated regions from class = annotationID | pick a random region instead
                    annotationID = 0

            if annotationID == 0: # select random region
                tree = None

                # select random patch - may include background
                if np.random.uniform() <= include_background_prob:
                    x_start = int(np.random.uniform(low=0, high=level_dims[0]-patch_width)*level_downsample)
                    y_start = int(np.random.uniform(low=0, high=level_dims[1]-patch_width)*level_downsample)


                # select random patch - does not include background
                else:
                    # track locations and vectorize mask
                    [y_ind, x_ind] = np.indices(np.shape(slide_mask))
                    y_ind = y_ind.ravel()
                    x_ind = x_ind.ravel()
                    mask_vec = slide_mask.ravel()

                    # select random pixel with tissue
                    idx = random.choice(np.argwhere(mask_vec==255))[0]
                    x_mask = x_ind[idx]
                    y_mask = y_ind[idx]

                    # calc wsi patch start indicies
                    x_start = int( ((x_mask / mask_scale) - (patch_width/2))*level_downsample)
                    y_start = int( ((y_mask / mask_scale) - (patch_width/2))*level_downsample)


            region = wsi.read_region((x_start,y_start), level, (patch_width,patch_width))
            mask = xml_to_mask(xml_path, (x_start,y_start), (patch_size,patch_size), tree=tree, downsample=downsample)

            if scale_factor > 1:
                region = region.resize((patch_size, patch_size), resample=1)
            region = np.array(region)[:,:,:3]

            if np.random.random() < augment:
                # augment image
                region, mask = augment_patch(region, mask)

            if augment > 0:
                # unpad
                region = region[2:-2,2:-2,:]
                mask = mask[2:-2,2:-2]

            return region, mask, x_start, y_start

        except: # wsi broken use a different level
            print('{} broken for level {}'.format(filename,level))
            level -= 1
            print('\ttrying level {}'.format(level))

def scale_patch(patch):
    # scale to [-1,1]
    patch = np.float32(patch)
    patch /= 127.5
    patch -= 1
    return patch

def get_grid_list(slide_path, patch_size, downsample, tile_step, wsi=None):
    # returns a list of (x,y) points where tissue is present

    slide_mask = get_slide_mask(slide_path, save_mask=False)
    # open slide once globally for efficency
    if wsi == None:
        wsi = large_image.getTileSource(slide_path)

    level_dims = get_slide_size(wsi=wsi)
    thumbnail_size = float(max(slide_mask.shape))
    mask_scale = thumbnail_size/max(level_dims)
    mask_patch_size = patch_size * mask_scale
    adj_patch_size = patch_size*downsample

    assert tile_step <= patch_size # step must less than patch_size

    # generate grid
    Xs = list(range(0,level_dims[0]-adj_patch_size,tile_step*downsample))
    Ys = list(range(0,level_dims[1]-adj_patch_size,tile_step*downsample))
    # manually add the last grid points (ensure it does not extend beyond the slide)
    Xs.append(level_dims[0] - adj_patch_size)
    Ys.append(level_dims[1] - adj_patch_size)

    points = []
    for X in Xs:
        for Y in Ys:
            Y_ = Y * mask_scale
            X_ = X * mask_scale
            if np.sum(slide_mask[int(Y_):int(Y_+mask_patch_size), int(X_):int(X_+mask_patch_size)]) > 0:
                points.append((X,Y))

    # calculate points size, offset
    p = np.array(points)
    x_pts = p[:,0]
    y_pts = p[:,1]
    min_x = min(x_pts)
    min_y = min(y_pts)
    max_x = max(x_pts)
    max_y = max(y_pts)
    tissue_offset = {'X':min_x, 'Y':min_y}
    tissue_size = [max_y-min_y, max_x-min_x]

    return points, len(points), tissue_offset, tissue_size

def get_slide_mask(filename, save_mask=True):
    # get or create wsi mask
    try:
        mask_path = '{}_MASK.png'.format(filename.decode().split('.')[0])
    except:
        mask_path = '{}_MASK.png'.format(filename.split('.')[0])

    # dont save mask, only return it
    if not save_mask:
        return save_wsi_thumbnail_mask(filename, save_mask=save_mask)

    # save mask
    if not os.path.isfile(mask_path):
        save_wsi_thumbnail_mask(filename, save_mask=save_mask)
    slide_mask = np.array(Image.open(mask_path))
    return slide_mask


def get_random_patch(wsi, l_dims, level, mask, patch_size, filename, downsample, include_background_prob, augment):
    return get_patch(wsi, None, 0, l_dims, level, mask, patch_size, filename, downsample, include_background_prob, augment)


def save_wsi_thumbnail_mask(filename, save_mask=True, thumbnail_size=2000):
    '''
    saves or returns a low resolution png mask of the tissue location in a WSI

    '''

    try: filename = filename.numpy()
    except: filename = filename
    wsi = large_image.getTileSource(filename)

    # def find_tissue_mask():
    #     thumbnail, _ = wsi.getThumbnail(width=thumbnail_size, height=thumbnail_size, format=large_image.tilesource.TILE_FORMAT_PIL)
    #     thumbnail_blurred = np.array(thumbnail.filter(ImageFilter.GaussianBlur(radius=10)))
    #     ret2,mask = cv2.threshold(thumbnail_blurred[:,:,0],0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    #     kernel = np.ones((5,5),np.uint8)
    #     mask = cv2.erode(mask,kernel,iterations = 1)
    #     mask[mask==0] = 1
    #     mask[mask==255] = 0
    #     return mask

    def find_tissue_mask():
        thumbnail, _ = wsi.getThumbnail(width=thumbnail_size, height=thumbnail_size, format=large_image.tilesource.TILE_FORMAT_PIL)
        t_array = np.array(thumbnail)[:,:,0] # red channel
        np.place(t_array, t_array==0, 255) # remove pure black
        thumbnail = Image.fromarray(t_array)
        thumbnail_blurred = np.array(thumbnail.filter(ImageFilter.GaussianBlur(radius=10))) # blur
        ret2,mask = cv2.threshold(thumbnail_blurred,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU) # threshold
        kernel = np.ones((5,5),np.uint8)
        mask = cv2.erode(mask,kernel,iterations = 1)
        mask[mask==0] = 1
        mask[mask==255] = 0
        return mask

    try:
        mask_path = '{}_MASK.png'.format(filename.decode().split('.')[0])
    except:
        mask_path = '{}_MASK.png'.format(filename.split('.')[0])

    # dont save mask, only return it
    if not save_mask:
        print('Creating tissue mask: [{}]'.format(mask_path))
        return find_tissue_mask()*255

    # save mask
    if not os.path.isfile(mask_path):
        print('Creating tissue mask: [{}]'.format(mask_path))
        mask = find_tissue_mask()*255
        mask_PIL = Image.fromarray(mask)
        mask_PIL.save(mask_path)

def augment_patch(region, mask):

    def colorshift(img, hbound=0.025, lbound=0.015): #Shift Hue of HSV space and Lightness of LAB space
        hShift=np.random.normal(0,hbound)
        lShift=np.random.normal(1,lbound)
        img=rgb2hsv(img)
        img[:,:,0]=(img[:,:,0]+hShift)
        img=hsv2rgb(img)
        img=rgb2lab(img)
        img[:,:,0]=exposure.adjust_gamma(img[:,:,0],lShift)
        img=lab2rgb(img)
        return img

    def PiecewiseAffine(img, mask, points=8):
        ### piecwise affine ###
        rows, cols = img.shape[0], img.shape[1]
        src_cols = np.linspace(0, cols, points)
        src_rows = np.linspace(0, rows, points)
        src_rows, src_cols = np.meshgrid(src_rows, src_cols)
        src = np.dstack([src_cols.flat, src_rows.flat])[0]
        # add offset
        dst_rows = np.zeros(src[:, 1].shape) + src[:, 1]
        for i in list(range(points))[1:-1]:
            dst_rows[i::points] += np.random.normal(loc=0, scale=rows/(points*10), size=dst_rows[i::points].shape)
        dst_cols = np.zeros(src[:, 0].shape) + src[:, 0]
        dst_cols[points:-points] += np.random.normal(loc=0,scale=rows/(points*10), size=dst_cols[points:-points].shape)
        dst = np.vstack([dst_cols, dst_rows]).T
        # compute transform
        tform = PiecewiseAffineTransform()
        tform.estimate(src, dst)
        # apply transform
        img = warp(img, tform, output_shape=(rows, cols))
        mask = warp(mask, tform, output_shape=(rows, cols))
        return img, mask

    region = (region/255.).astype(np.float64)
    region = colorshift(region)
    region, mask = PiecewiseAffine(region, mask)
    region = np.uint8(region*255.)

    return region, mask


def get_slide_label(filename, data_label_xlsx):
    data_label_xlsx = str(data_label_xlsx.numpy())
    filename = str(filename.numpy())
    # get slide label
    df = pd.read_excel(data_label_xlsx)

    try:
        name = filename.decode().split('/')[-1]
    except:
        name = filename.split('/')[-1]

    index = df.index[df['wsi']==name].tolist()
    if index == []:
        label = np.array([-1])
    else:
        label = np.array(df['class'][index])
    return label

def get_slide_size(filename=None, wsi=None):
    if filename:
        wsi = large_image.getTileSource(filename)

    width = wsi.sizeX
    height = wsi.sizeY
    slide_size = (width, height)
    return slide_size
