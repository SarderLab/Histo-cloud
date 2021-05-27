from glob import glob
import numpy as np
import sys
import lxml.etree as ET
import os
import girder_client
import argparse
from xml_to_json import convert_xml_json
import json

parser = argparse.ArgumentParser()
parser.add_argument("username", help="DSA username")
parser.add_argument("password", help="DSA password") # credentials for server access
parser.add_argument("DSA_folder", help="DSA folder containing corresponding slides") # local path example: brendonl/data folder
parser.add_argument("xml_folder", help="Local folder containing xml annotations") # local path to xmls
parser.add_argument("names", help='delimited list input of annotation layer names | example <name1,name2>', type=str)
parser.add_argument("-ext", "--extension", help="wsi file extension")
parser.add_argument("-c", "--collection", help="the data is in a collection not a user")

args = parser.parse_args()

if args.extension: ext = args.extension
else: ext = '.svs'

gc = girder_client.GirderClient(apiUrl='https://athena.ccr.buffalo.edu/api/v1')
gc.authenticate(args.username,args.password)

xmls = glob('{}/*.xml'.format(args.xml_folder))
print('\nfound [ {} ] xml annotations in [ {} ]\n'.format(len(xmls), args.xml_folder))
print('using layer names: [ {} ]'.format(args.names))


# get girder ids
for xml in xmls:
    slidename = os.path.basename(xml)
    slidename = os.path.splitext(slidename)[0] + ext

    try:
        if args.collection:
            g_path = '/collection/{}/{}'.format(args.DSA_folder, slidename)
            _id = gc.resourceLookup(g_path)['_id']
        else:
            g_path = '/user/{}/{}'.format(args.DSA_folder, slidename)
            _id = gc.resourceLookup(g_path)['_id']
        print('found: [ {} ] [ {} ]'.format(slidename, _id))

    except:
        print('!!! [ {} ] not found !!!'.format(g_path))
        continue

    # convert xml to json
    tree = ET.parse(xml)
    root = tree.getroot()
    names = args.names.split(',')
    annot = convert_xml_json(root, names)

    _ = gc.post('/annotation/item/{}'.format(_id),data=json.dumps(annot))
    print('annotation uploaded...\n')
