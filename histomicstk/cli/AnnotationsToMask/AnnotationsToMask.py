import os

from histomicstk.annotations_and_masks.annotation_and_mask_utils import (
    get_bboxes_from_slide_annotations, _get_idxs_for_all_rois)
from histomicstk.annotations_and_masks.annotations_to_masks_handler import (
    get_roi_mask)

from pandas import read_csv
from imageio import imwrite

import skimage.io
import large_image
from histomicstk.cli import utils
from histomicstk.cli.utils import CLIArgumentParser
import json

def get_all_roi_masks_for_slide(
        input_img, input_ann, GTCODE_PATH, MASK_SAVEPATH, slide_name=None,
        verbose=True, monitorPrefix="", get_roi_mask_kwargs=dict()):
    """Parse annotations and saves ground truth masks for ALL ROIs.
    Get all ROIs in a single slide. This is a wrapper around get_roi_mask()
    which should be referred to for implementation details.
    Parameters
    -----------
    input_img : object
        input large image object
    input_ann : object
        input annotation object
    GTCODE_PATH : str
        path to the ground truth codes and information
        csv file. Refer to the docstring of get_roi_mask() for more info.
    MASK_SAVEPATH : str
        path to directory to save ROI masks
    slide_name (optional) : str
        If not given, it's inferred using a server request using girder client.
    verbose (optional) : bool
        Print progress to screen?
    monitorPrefix (optional) : str
        text to prepend to printed statements
    get_roi_mask_kwargs : dict
        extra kwargs for get_roi_mask()
    Returns
    --------
    list of strs
        save paths for ROIs
    """
    # read ground truth codes and information
    GTCodes = read_csv(GTCODE_PATH)
    GTCodes.index = GTCodes.loc[:, 'group']
    if any(GTCodes.loc[:, 'GT_code'] <= 0):
        raise Exception("All GT_code must be > 0")

    # get annotations for slide
    slide_annotations = input_ann

    # get bounding box information for all annotations
    element_infos = get_bboxes_from_slide_annotations(slide_annotations)

    # get indices of rois
    idxs_for_all_rois = _get_idxs_for_all_rois(
        GTCodes=GTCodes, element_infos=element_infos)

    savenames = []

    for roino, idx_for_roi in enumerate(idxs_for_all_rois):

        roicountStr = "%s: roi %d of %d" % (
            monitorPrefix, roino + 1, len(idxs_for_all_rois))

        # get roi mask and info
        ROI, roiinfo = get_roi_mask(
            slide_annotations=slide_annotations, element_infos=element_infos,
            GTCodes_df=GTCodes.copy(), idx_for_roi=idx_for_roi,
            monitorPrefix=roicountStr, **get_roi_mask_kwargs)

        ## make directory for the mask
        MASK_SAVEPATH_MASK = MASK_SAVEPATH + '/mask'
        # create folders if necessary
        for folder in [MASK_SAVEPATH_MASK, ]:
            try:
                os.mkdir(folder)
            except:
                pass

        # now save roi
        ROINAMESTR = "%s_left-%d_top-%d" % (
            slide_name, roiinfo['XMIN'], roiinfo['YMIN'])
        savename = os.path.join(MASK_SAVEPATH_MASK, ROINAMESTR + ".png")
        if verbose:
            print("%s: Saving %s\n" % (roicountStr, savename))
        imwrite(im=ROI, uri=savename)

        region = [roiinfo['XMIN'], roiinfo['YMIN'], roiinfo['BBOX_WIDTH'], roiinfo['BBOX_HEIGHT']]
        maxRegionSize = 5000

        ## make directory for the region
        MASK_SAVEPATH_REG = MASK_SAVEPATH + '/region'
        # create folders if necessary
        for folder in [MASK_SAVEPATH_REG, ]:
            try:
                os.mkdir(folder)
            except:
                pass

        #######save images
        im_input = input_img.getRegion(
            format=large_image.tilesource.TILE_FORMAT_NUMPY,
            **utils.get_region_dict(region, maxRegionSize, input_img))[0]

        ROINAMESTR1 = "%s_left-%d_top-%d" % (
            slide_name, roiinfo['XMIN'], roiinfo['YMIN'])
        savename1 = os.path.join(MASK_SAVEPATH_REG, ROINAMESTR1 + ".png")
        skimage.io.imsave(savename1, im_input)
        if verbose:
            print("%s: Saving %s\n" % (roicountStr, savename1))

        savenames.append(savename)

    return savenames


def main(args):
    #
    # read input image
    #
    print('>> Reading input image')

    print(args.inputImageFile)
    slide_name = os.path.basename(args.inputImageFile)
    slide_name = os.path.splitext(slide_name)[0]

    ts = large_image.getTileSource(args.inputImageFile)

    #
    # read annotation file
    #
    print('\n>> Reading annotation file ...\n')

    with open(args.inputAnnotationFile) as json_file:
        annotation_data = json.load(json_file)

    data = []
    if isinstance(annotation_data, dict):
        data.append(annotation_data)

    elif isinstance(annotation_data, list):
        data = list(annotation_data)

    #
    # read GTCode file
    #
    print('\n>> Reading Ground Truth file ...\n')

    GTCodes = args.inputGTCodeFile

    #
    #  convert annotation to mask
    #
    print('\n>> Performing conversion from annotation to mask ...\n')

    MASK_SAVEPATH = args.outputDirectory
    # create folders if necessary
    for folder in [MASK_SAVEPATH, ]:
        try:
            os.mkdir(folder)
        except:
            pass

    get_all_roi_masks_for_slide(ts, data, GTCodes,
                                MASK_SAVEPATH=MASK_SAVEPATH, slide_name=slide_name, verbose=True,
                                get_roi_mask_kwargs={
                                    'iou_thresh': 0.0, 'crop_to_roi': True, 'use_shapely': True,
                                    'verbose': True},
                                )



if __name__ == "__main__":
    main(CLIArgumentParser().parse_args())
