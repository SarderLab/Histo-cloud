import girder_client, os, json, tempfile, zipfile
import xml.etree.ElementTree as ET
import numpy as np
from histomicstk.cli.utils import CLIArgumentParser
from glob import glob

import sys
sys.path.append("..")
from deeplab.utils.xml_to_json import convert_xml_json

def main(args):

    # get compartments
    compartments = args.classes

    # get folder
    folder = args.inputFolder
    girder_folder_id = folder.split('/')[-2]
    _ = os.system("printf '\nUsing data from girder_client Folder: {}\n'".format(folder))

    os.system("ls -lh '{}'".format(folder))
    gc = girder_client.GirderClient(apiUrl=args.girderApiUrl)
    gc.authenticate(apiKey=args.apiKey)
    # get files in folder

    _ = os.system("printf '\n---\n\ngirder folder: [{}]\n'".format(girder_folder_id))
    files = list(gc.listItem(girder_folder_id))

    # dict to link filename to gc id
    item_dict = {}
    for file in files:
        d = {file['name']:file['_id']}
        item_dict.update(d)

    # download slides and annotations to tmp directory
    for file in files:
        xmlname = file['name']

        if not '.xml' in xmlname:
            continue

        _ = os.system("printf '\n---\n\nFOUND: [{}]\n'".format(xmlname))

        # get slide name
        matches = glob('{}/{}.*'.format(folder, os.path.splitext(xmlname)[0]))
        slidename = None
        for match in matches:
            if not '.xml' in match:
                slidename = match
                break

        if slidename == None:
            _ = os.system("printf '\tSlide not found for [{}] annotation\n'".format(xmlname))
            continue

        # create json from xml
        tree = ET.parse('{}/{}'.format(folder,xmlname))
        root = tree.getroot()
        annot = convert_xml_json(root, compartments)

        # put annotation
        id = item_dict[os.path.basename(slidename)]
        item = gc.getItem(id)
        _ = gc.post('/annotation/item/{}'.format(item['_id']),data=json.dumps(annot))
        _ = os.system("printf '\tannotations created for [{}]\n'".format(slidename))


if __name__ == "__main__":
    main(CLIArgumentParser().parse_args())
