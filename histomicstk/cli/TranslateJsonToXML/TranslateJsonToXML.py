import json
import xml.etree.ElementTree as ET
from histomicstk.cli.utils import CLIArgumentParser

import logging
logging.basicConfig()


def indent(elem, level=0):

    i = "\n" + level * "  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for elem in elem:
            indent(elem, level + 1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i


def convert_json_xml(data):

    color_list = [65280, 13691392, 55039, 16746546, 65521, 5406802, 16776960, 8421504, 255, 26112, 16711680,
                  45568, 46591, 16734208, 11963392]

    root = ET.Element("Annotations", MicronsPerPixel="")

    if isinstance(data, dict):
        # print("dict")
        ele = None
        layer_name = None
        if 'annotation' in data:
            ann = data['annotation']
            ele = ann['elements']
            layer_name = ann['name']
        elif 'elements' in data:
            ele = data['elements']
            layer_name = data['name']

        annotation = ET.SubElement(root, "Annotation", Id="1", Name="", ReadOnly="0", NameReadOnly="0",
                                   LineColorReadOnly="0",
                                   Incremental="0", Type="4", LineColor=str(color_list[0]), Visible="1",
                                   Selected="1", MarkupImagePath="", MacroName="")
        attributes = ET.SubElement(annotation, "Attributes")
        ET.SubElement(attributes, "Attribute", Name=layer_name, Id="0", Value="")
        regions = ET.SubElement(annotation, "Regions")

        regionAttributeHeaders = ET.SubElement(regions, "RegionAttributeHeaders")
        ET.SubElement(regionAttributeHeaders, "AttributeHeader", Id="9999", Name="Region", ColumnWidth="-1")
        ET.SubElement(regionAttributeHeaders, "AttributeHeader", Id="9997", Name="Length", ColumnWidth="-1")
        ET.SubElement(regionAttributeHeaders, "AttributeHeader", Id="9996", Name="Area", ColumnWidth="-1")
        ET.SubElement(regionAttributeHeaders, "AttributeHeader", Id="9998", Name="Text", ColumnWidth="-1")
        ET.SubElement(regionAttributeHeaders, "AttributeHeader", Id="1", Name="Description", ColumnWidth="-1")
        reg = []
        for j, region in enumerate(ele, start=1):
            reg.append(ET.SubElement(regions, "Region", Id=str(j), Type="1", Zoom="", Selected="0",
                                     ImageLocation="",
                                     ImageFocus="-1", Length="", Area="", LengthMicrons="", AreaMicrons="",
                                     Text="",
                                     NegativeROA="0", InputRegionId="0", Analyze="1", DisplayId=str(j)))
            vertices = ET.SubElement(reg[j - 1], "Vertices")
            po = region['points']
            ver = []
            for k in po:
                ver.append(ET.SubElement(vertices, "Vertex", X=str(int(k[0])), Y=str(int(k[1])),
                                         Z="0"))

    elif isinstance(data, list):
        # print("list")
        for i, ann_layer in enumerate(data, start=1):
            ann = ann_layer['annotation']
            ele = ann['elements']
            layer_name = ann['name']
            annotation = ET.SubElement(root, "Annotation", Id=str(i), Name="", ReadOnly="0", NameReadOnly="0",
                                       LineColorReadOnly="0",
                                       Incremental="0", Type="4", LineColor=str(color_list[(i - 1) % 15]), Visible="1",
                                       Selected="1", MarkupImagePath="", MacroName="")
            attributes = ET.SubElement(annotation, "Attributes")
            ET.SubElement(attributes, "Attribute", Name=layer_name, Id="0", Value="")
            regions = ET.SubElement(annotation, "Regions")

            regionAttributeHeaders = ET.SubElement(regions, "RegionAttributeHeaders")
            ET.SubElement(regionAttributeHeaders, "AttributeHeader", Id="9999", Name="Region", ColumnWidth="-1")
            ET.SubElement(regionAttributeHeaders, "AttributeHeader", Id="9997", Name="Length", ColumnWidth="-1")
            ET.SubElement(regionAttributeHeaders, "AttributeHeader", Id="9996", Name="Area", ColumnWidth="-1")
            ET.SubElement(regionAttributeHeaders, "AttributeHeader", Id="9998", Name="Text", ColumnWidth="-1")
            ET.SubElement(regionAttributeHeaders, "AttributeHeader", Id="1", Name="Description", ColumnWidth="-1")
            reg = []
            for j, region in enumerate(ele, start=1):
                reg.append(ET.SubElement(regions, "Region", Id=str(j), Type="1", Zoom="", Selected="0",
                                         ImageLocation="",
                                         ImageFocus="-1", Length="", Area="", LengthMicrons="", AreaMicrons="",
                                         Text="",
                                         NegativeROA="0", InputRegionId="0", Analyze="1", DisplayId=str(j)))
                vertices = ET.SubElement(reg[j - 1], "Vertices")
                po = region['points']
                ver = []
                for k in po:
                    ver.append(ET.SubElement(vertices, "Vertex", X=str(int(k[0])), Y=str(int(k[1])),
                                             Z="0"))

    else:
        raise ValueError('Check the format of json file')

    indent(root)
    return root


def main(args):
    #
    # read annotation file
    #
    print('\n>> Loading annotation file ...\n')

    with open(args.inputAnnotationFile) as json_file:
        annotation_data = json.load(json_file)

    #
    #  convert json to xml
    #
    print('\n>> Performing conversion ...\n')
    result_data = convert_json_xml(annotation_data)

    #
    # write annotation xml file
    #
    print('\n>> Writing xml file ...\n')
    tree = ET.ElementTree(result_data)
    tree.write(args.outputAnnotationFile)


if __name__ == "__main__":
    main(CLIArgumentParser().parse_args())
