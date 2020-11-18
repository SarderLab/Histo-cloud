import json, os
import xml.etree.ElementTree as ET
from histomicstk.cli.utils import CLIArgumentParser

import logging
logging.basicConfig()


def convert_xml_json(root, names):

    colorList = ["rgb(0, 255, 128)", "rgb(0, 255, 255)", "rgb(255, 255, 0)", "rgb(255, 128, 0)", "rgb(0, 128, 255)",
                 "rgb(0, 0, 255)", "rgb(0, 102, 0)", "rgb(153, 0, 0)", "rgb(0, 153, 0)", "rgb(102, 0, 204)",
                 "rgb(76, 216, 23)", "rgb(102, 51, 0)", "rgb(128, 128, 128)", "rgb(0, 153, 153)", "rgb(0, 0, 0)"]

    anns = root.findall('Annotation')
    print(len(anns))
    print(anns)
    print(len(names))
    print(names)
    assert len(anns) == len(names)

    data = []
    for n, child in enumerate(anns):
        dataDict = dict()
        name = names[n]
        _ = os.system("printf 'Building JSON layer: [{}]\n'".format(name))
        element = []
        reg = child.find('Regions')
        for i in reg.findall('Region'):
            eleDict = dict()
            eleDict["closed"] = True

            lineColor = colorList[n % 15]
            eleDict["lineColor"] = lineColor

            fillColor = lineColor[:3]+'a'+lineColor[3:-1]+', 0.5)'
            eleDict["fillColor"] = fillColor

            eleDict["lineWidth"] = 2
            points = []
            ver = i.find('Vertices')
            for j in ver.findall('Vertex'):
                eachPoint = []
                eachPoint.append(float(j.get('X')))
                eachPoint.append(float(j.get('Y')))
                eachPoint.append(float(j.get('Z')))
                points.append(eachPoint)
            eleDict["points"] = points
            eleDict["type"] = "polyline"
            element.append(eleDict)
        dataDict["elements"] = element
        dataDict["name"] = name
        data.append(dataDict)

    return data




def main(args):
    #
    # read annotation file
    #
    print('\n>> Loading annotation file ...\n')

    tree = ET.parse(args.inputAnnotationFile)
    root = tree.getroot()

    names = args.classes

    #
    #  convert json to xml
    #
    print('\n>> Performing conversion ...\n')
    annotation = convert_xml_json(root, names)

    #
    # write annotation xml file
    #
    print('\n>> Writing xml file ...\n')
    with open(args.outputAnnotationFile, 'w') as annotation_file:
        json.dump(annotation, annotation_file, indent=2, sort_keys=False)


if __name__ == "__main__":
    main(CLIArgumentParser().parse_args())
