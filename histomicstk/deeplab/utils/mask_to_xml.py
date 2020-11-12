import os
import cv2
import numpy as np
from PIL import Image
import lxml.etree as ET

"""
xml_path (string) - the filename of the saved xml
mask (array) - the mask to convert to xml - uint8 array
downsample (int) - amount of downsampling done to the mask
                    points are upsampled - this can be used to simplify the mask
min_size_thresh (int) - the minimum objectr size allowed in the mask. This is referenced from downsample=1
xml_color (list) - list of binary color values to be used for classes

"""

def mask_to_xml(xml_path, mask, downsample=1, min_size_thresh=0, simplify_contours=0, xml_color=[65280, 65535, 33023, 255, 16711680], verbose=0, return_root=False):

    min_size_thresh /= downsample

    # create xml tree
    Annotations = xml_create()

    # get all classes
    classes = np.unique(mask)

    # add annotation classes to tree
    for class_ in range(max(classes)+1)[1:]:
        if verbose:
            print('Creating class: [{}]'.format(class_))
        Annotations = xml_add_annotation(Annotations=Annotations, xml_color=xml_color, annotationID=class_)

    # add contour points to tree classwise
    for class_ in classes[1:]: # iterate through all classes

        if verbose:
            print('Working on class [{} of {}]'.format(class_, max(classes)))

        # binarize the mask w.r.t. class_
        binaryMask = mask==class_

        # get contour points of the mask
        pointsList = get_contour_points(binaryMask, downsample=downsample, min_size_thresh=min_size_thresh, simplify_contours=simplify_contours)
        for i in range(np.shape(pointsList)[0]):
            pointList = pointsList[i]
            Annotations = xml_add_region(Annotations=Annotations, pointList=pointList, annotationID=class_)

    if return_root:
        # return root, do not save xml file
        return Annotations

    # save the final xml file
    xml_save(Annotations=Annotations, filename='{}.xml'.format(xml_path.split('.')[0]))


def get_contour_points(mask, downsample, min_size_thresh=0, simplify_contours=0, offset={'X': 0,'Y': 0}):
    # returns a dict pointList with point 'X' and 'Y' values
    # input greyscale binary image
    #_, maskPoints, contours = cv2.findContours(np.array(mask), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
    maskPoints, contours = cv2.findContours(np.uint8(mask), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)

    # remove small regions
    too_small = []
    for idx, cnt in enumerate(maskPoints):
        area = cv2.contourArea(cnt)
        if area < min_size_thresh:
            too_small.append(idx)
    if too_small != []:
        too_small.reverse()
        for idx in too_small:
            maskPoints.pop(idx)

    if simplify_contours > 0:
        for idx, cnt in enumerate(maskPoints):
            epsilon = simplify_contours*cv2.arcLength(cnt,True)
            approx = cv2.approxPolyDP(cnt,epsilon,True)
            maskPoints[idx] = approx

    pointsList = []
    for j in range(np.shape(maskPoints)[0]):
        pointList = []
        for i in range(0,np.shape(maskPoints[j])[0]):
            point = {'X': (maskPoints[j][i][0][0] * downsample) + offset['X'], 'Y': (maskPoints[j][i][0][1] * downsample) + offset['Y']}
            pointList.append(point)
        pointsList.append(pointList)
    return pointsList

### functions for building an xml tree of annotations ###
def xml_create(): # create new xml tree
    # create new xml Tree - Annotations
    Annotations = ET.Element('Annotations')
    return Annotations

def xml_add_annotation(Annotations, xml_color, annotationID=None): # add new annotation
    # add new Annotation to Annotations
    # defualts to new annotationID
    if annotationID == None: # not specified
        annotationID = len(Annotations.findall('Annotation')) + 1
    Annotation = ET.SubElement(Annotations, 'Annotation', attrib={'Type': '4', 'Visible': '1', 'ReadOnly': '0', 'Incremental': '0', 'LineColorReadOnly': '0', 'LineColor': str(xml_color[annotationID-1]), 'Id': str(annotationID), 'NameReadOnly': '0'})
    Regions = ET.SubElement(Annotation, 'Regions')
    return Annotations

def xml_add_region(Annotations, pointList, annotationID=-1, regionID=None): # add new region to annotation
    # add new Region to Annotation
    # defualts to last annotationID and new regionID
    Annotation = Annotations.find("Annotation[@Id='" + str(annotationID) + "']")
    Regions = Annotation.find('Regions')
    if regionID == None: # not specified
        regionID = len(Regions.findall('Region')) + 1
    Region = ET.SubElement(Regions, 'Region', attrib={'NegativeROA': '0', 'ImageFocus': '-1', 'DisplayId': '1', 'InputRegionId': '0', 'Analyze': '0', 'Type': '0', 'Id': str(regionID)})
    Vertices = ET.SubElement(Region, 'Vertices')
    for point in pointList: # add new Vertex
        ET.SubElement(Vertices, 'Vertex', attrib={'X': str(point['X']), 'Y': str(point['Y']), 'Z': '0'})
    # add connecting point
    ET.SubElement(Vertices, 'Vertex', attrib={'X': str(pointList[0]['X']), 'Y': str(pointList[0]['Y']), 'Z': '0'})
    return Annotations

def xml_save(Annotations, filename):
    xml_data = ET.tostring(Annotations, pretty_print=True)
    #xml_data = Annotations.toprettyxml()
    f = open(filename, 'w')
    f.write(xml_data.decode())
    f.close()

def read_xml(filename):
    # import xml file
    tree = ET.parse(filename)
    root = tree.getroot()

if __name__ == '__main__':
    main()
