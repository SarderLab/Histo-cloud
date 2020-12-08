import girder_client, os, json, tempfile, zipfile
import xml.etree.ElementTree as ET
import numpy as np
from histomicstk.cli.utils import CLIArgumentParser
from glob import glob

import sys
sys.path.append("..")
from deeplab.utils.mask_to_xml import xml_create, xml_add_annotation, xml_add_region, xml_save
from deeplab.utils.xml_to_mask import write_minmax_to_xml

def main(args):

    # get compartments
    compartments = args.classes

    _ = os.system("printf '\n\n---\n\nUsing annotated layers: {}'".format(compartments))

    num_colors = len(compartments)
    xml_color_lst=[65280, 65535, 33023, 255, 16711680] # for conversion to xml
    q, r = divmod(num_colors, len(xml_color_lst))
    xml_color = q * xml_color_lst + xml_color_lst[:r]

    # get folder
    folder = args.inputFolder
    girder_folder_id = folder.split('/')[-2]
    _ = os.system("printf 'Using data from girder_client Folder: {}\n'".format(folder))

    os.system("ls -lh '{}'".format(folder))

    # create tmp directory for storing intemediate files
    tmp = '{}/xml_directory/'.format(folder)
    os.mkdir(tmp)

    cwd = os.getcwd()

    gc = girder_client.GirderClient(apiUrl=args.girderApiUrl)
    gc.setToken(args.girderToken)
    # get files in folder
    files = gc.listItem(girder_folder_id)

    # download slides and annotations to tmp directory
    slides_used = []
    for file in files:
        slidename = file['name']
        _ = os.system("printf '\n---\n\nFOUND: [{}]\n'".format(slidename))
        skipSlide = 0

        # get annotation
        item = gc.getItem(file['_id'])
        annot = gc.get('/annotation/item/{}'.format(item['_id']), parameters={'sort': 'updated'})
        annot.reverse()
        _ = os.system("printf '\tfound [{}] annotation layers...\n'".format(len(annot)))

        # create root for xml file
        xmlAnnot = xml_create()

        # all compartments
        for class_,compart in enumerate(compartments):
            compart = compart.replace(' ','')
            class_ +=1
            # add layer to xml
            xmlAnnot = xml_add_annotation(Annotations=xmlAnnot, xml_color=xml_color, annotationID=class_)

            # test all annotation layers in order created
            for iter,a in enumerate(annot):

                try:
                    # check for annotation layer by name
                    a_name = a['annotation']['name'].replace(' ','')
                except:
                    a_name = None

                if a_name == compart:
                    # track all layers present
                    skipSlide +=1

                    pointsList = []

                    # load json data
                    _ = os.system("printf '\tloading annotation layer: [{}]\n'".format(compart))

                    a_data = a['annotation']['elements']

                    for data in a_data:
                        pointList = []
                        points = data['points']
                        for point in points:
                            pt_dict = {'X': round(point[0]), 'Y': round(point[1])}
                            pointList.append(pt_dict)
                        pointsList.append(pointList)

                    # write annotations to xml
                    for i in range(np.shape(pointsList)[0]):
                        pointList = pointsList[i]
                        xmlAnnot = xml_add_region(Annotations=xmlAnnot, pointList=pointList, annotationID=class_)

                    # print(a['_version'], a['updated'], a['created'])
                    break

        if skipSlide != len(compartments):
            _ = os.system("printf '\tThis slide is missing annotation layers\n'")
            _ = os.system("printf '\tSKIPPING SLIDE...\n'")
            del xmlAnnot
            continue # correct layers not present


        # include slide and fetch annotations
        _ = os.system("printf '\tUSING SLIDE...\n'")

        # save the final xml file
        xml_path = '{}/{}.xml'.format(tmp, os.path.splitext(slidename)[0])
        _ = os.system("printf '\tsaving a created xml annotation file: [{}]\n'".format(xml_path))
        xml_save(Annotations=xmlAnnot, filename=xml_path)
        write_minmax_to_xml(xml_path) # to avoid trying to write to the xml from multiple workers
        del xmlAnnot

    # move model to zipped file for output
    os.listdir(tmp)
    os.chdir(tmp)
    os.system('pwd')
    os.system('ls -lh')

    # get xmls
    filelist = glob('*.xml')
    # zip xmls into new folder
    z = zipfile.ZipFile(args.output_folder, 'w')
    for file in filelist:
        z.write(file, compress_type=zipfile.ZIP_DEFLATED)
    z.close()


if __name__ == "__main__":
    main(CLIArgumentParser().parse_args())
