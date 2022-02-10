from glob import glob
import numpy as np
import sys
import os
import girder_client
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument("username", help="DSA username")
parser.add_argument("password", help="DSA password") # credentials for server access
parser.add_argument("DSA_folder", help="DSA folder containing corresponding slides") # local path example: brendonl/data folder
parser.add_argument("names", help='delimited list input of annotation layer names | example <name1,name2>', type=str)
parser.add_argument("-ext", "--extension", help="wsi file extension")
parser.add_argument("-c", "--collection", help="the data is in a collection not a user")

args = parser.parse_args()

if args.extension: ext = args.extension
else: ext = '.svs'

gc = girder_client.GirderClient(apiUrl='<URL>/api/v1')
gc.authenticate(args.username,args.password)

remove_names = (args.names).split(',')

print('removing layer names: {}'.format(remove_names))

# get girder ids
try:
    if args.collection:
        g_path = '/collection/{}'.format(args.DSA_folder)
        _id = gc.resourceLookup(g_path)['_id']
    else:
        g_path = '/user/{}'.format(args.DSA_folder)
        _id = gc.resourceLookup(g_path)['_id']
    print(f'found folder with girder id: [{_id}]\n')

except:
    print('!!! [ {} ] not found !!!'.format(g_path))
    exit()
    
# get files in folder
files = gc.listItem(_id)

for file in files:
    slidename = file['name']
    print(f'\nworking on: [{slidename}]')
    
    # get annotation
    item = gc.getItem(file['_id'])
    annot = gc.get('/annotation/item/{}'.format(item['_id']), parameters={'sort': 'updated'})
    annot.reverse()
    annot = list(annot)

    # test all annotation layers in order created
    for iter,a in enumerate(annot):

        try:
            # check for annotation layer by name
            a_name = a['annotation']['name']
        except:
            a_name = None

        if a_name in remove_names:
            annot_id = a['_id']
            # remove layer
            print(f'\tremoving layer: [{a_name}]')
            _ = gc.delete('/annotation/{}'.format(annot_id))
            
