import girder_client, os, shutil, json, time, tempfile, zipfile, cv2
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
from termcolor import colored
from histomicstk.cli.utils import CLIArgumentParser
from glob import glob
from large_image.cache_util import cachesClear

import sys
sys.path.append("..")
from deeplab.utils.mask_to_xml import xml_create, xml_add_annotation, xml_add_region, xml_save
from deeplab.utils.xml_to_mask import write_minmax_to_xml, xml_to_mask
import large_image

def main(args):

    # get anot_layers
    anot_layers = args.classes

    _ = os.system("printf '\n\n---\n\nUsing annotated layer: {}'".format(anot_layers))

    # get folder
    folder = args.inputFolder
    girder_folder_id = folder.split('/')[-2]
    _ = os.system("printf 'Using data from girder_client Folder: {}\n'".format(folder))

    os.system("ls -lh '{}'".format(folder))

    # create tmp directory for storing intemediate files
    tmp = '{}/tmp_file_dir/'.format(folder)
    os.mkdir(tmp)

    # authenticate girder client
    gc = girder_client.GirderClient(apiUrl=args.girderApiUrl)
    gc.setToken(args.girderToken)
    # get files in folder
    files = gc.listItem(girder_folder_id)

    # create feature dict
    all_features = {}
    for compart in anot_layers:
        all_features[compart] = []

    # get features for all annotated slides
    for file_iter, file in enumerate(files):
        slidename = file['name']
        _ = os.system("printf '\n---\n\nFOUND: [{}]\n'".format(slidename))

        df = pd.read_excel(args.slide_labels)
        slide_label = df.loc[df.iloc[:,0] == slidename]
        if not slide_label.empty:
            label = slide_label.iloc[0,1:].to_dict()
        else:
            label = None

        # get annotation
        item = gc.getItem(file['_id'])
        annot = gc.get('/annotation/item/{}'.format(item['_id']), parameters={'sort': 'updated'})
        annot.reverse()
        if len(annot) > 0:
            _ = os.system("printf '\tfound [{}] annotation layers...\n'".format(len(annot)))

        # all anot_layers
        for class_,compart in enumerate(anot_layers):
            features = []
            compart = compart.replace(' ','')

            # test all annotation layers in order created
            for iter,a in enumerate(annot):

                try:
                    # check for annotation layer by name
                    a_name = a['annotation']['name'].replace(' ','')
                except:
                    a_name = None

                if a_name == compart:
                    pointsList = []

                    # load json data
                    _ = os.system("printf '\textracting features from: [{}]\n'".format(compart))

                    a_data = a['annotation']['elements']

                    # iterate through annotations
                    for data in a_data:
                        pointList = []
                        points = np.array(data['points'])
                        points = np.round(points[:,:2])

                        # get bounds of points
                        x_min = int(min(points[:,0]))
                        x_max = int(max(points[:,0]))
                        y_min = int(min(points[:,1]))
                        y_max = int(max(points[:,1]))

                        # remove offset
                        points -=[x_min,y_min]
                        points = np.int32(points)

                        # extract image region
                        wsi = large_image.getTileSource('{}/{}'.format(folder, slidename))

                        # assert the annotation is in the slide bounds and not width 0
                        try:
                            assert x_min >= 0
                            assert y_min >= 0
                            assert x_max <= wsi.sizeX
                            assert y_max <= wsi.sizeY
                            assert y_max-y_min > 0
                            assert x_max-x_min > 0
                        except:
                            continue

                        # set native mag if None
                        mm_x = wsi.getNativeMagnification()['mm_x']
                        if mm_x is None:
                            mm_x = 0.0005
                            wsi.getNativeMagnification = lambda: {"magnification": 40, "mm_x": mm_x, "mm_y": mm_x}

                        # extract region
                        region, _ = wsi.getRegion(region=dict(left=x_min,top=y_min, right=x_max, bottom=y_max, units='base_pixels'), format=large_image.tilesource.TILE_FORMAT_NUMPY)

                        # create mask
                        mask = np.zeros([y_max-y_min, x_max-x_min], dtype=np.uint8)
                        cv2.fillPoly(mask, [points], 1)

                        if args.use_microns:
                            microns = mm_x*1000 # microns/pixel
                        else:
                            microns = None

                        feats = compute_image_contour_features(img=region, mask=mask, cnt=points, filename=slidename, file_iter=file_iter, microns=microns)
                        if feats is not None:
                            if label is not None:
                                feats.update(label)
                            features.append(feats)

                    break

            all_features[anot_layers[class_]].extend(features)

        # clear large image caches
        cachesClear()

    # create excel file from feature array
    _ = os.system("printf '\tWriting Excel file: [{}]\n'".format(args.output_filename))
    with pd.ExcelWriter(args.output_filename) as writer:
        for compart in list(all_features.keys()):
            df = pd.DataFrame(all_features[compart])
            df.to_excel(writer, index=False, sheet_name=compart)


def compute_image_contour_features(img,mask,cnt,filename,file_iter,microns=None):
    feats = {}

    feats['file number'] = file_iter
    feats['filename'] = filename

    ###########################################################################
    ### Contour features ###
    ###########################################################################

    # area
    area = cv2.contourArea(cnt)
    if microns is None:
        feats['area (pixels^2)'] = area
    else:
        feats['area (microns^2)'] = area*(microns**2)

    # aspect ratio
    x,y,w,h = cv2.boundingRect(cnt)
    aspect_ratio = float(w)/h
    feats['aspect ratio'] = aspect_ratio

    # Extent
    rect_area = w*h
    extent = float(area)/rect_area
    feats['extent'] = extent

    # solidity
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    try:
        solidity = float(area)/hull_area
        feats['solidity'] = solidity
    except: pass

    # perimiter
    perimeter = cv2.arcLength(cnt,True)
    if microns is None:
        feats['perimeter (pixels)'] = perimeter
    else:
         feats['perimeter (microns)'] = perimeter*microns

    ###########################################################################
    ## Image features
    ###########################################################################

    flat_mask = mask.flatten()

    # red channel feats
    channel = img[:,:,0]
    channel = channel.flatten()
    channel = channel[flat_mask==1]
    feats['red channel average'] = channel.mean()
    feats['red channel std'] = np.std(channel)

    # green channel feats
    channel = img[:,:,1]
    channel = channel.flatten()
    channel = channel[flat_mask==1]
    feats['green channel average'] = channel.mean()
    feats['green channel std'] = np.std(channel)

    # blue channel feats
    channel = img[:,:,2]
    channel = channel.flatten()
    channel = channel[flat_mask==1]
    feats['blue channel average'] = channel.mean()
    feats['blue channel std'] = np.std(channel)

    ###########################################################################
    ##
    ###########################################################################

    return feats


if __name__ == "__main__":
    main(CLIArgumentParser().parse_args())
