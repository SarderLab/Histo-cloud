import girder_client, os, shutil, json, time, tempfile, zipfile
import xml.etree.ElementTree as ET
import numpy as np
from termcolor import colored
from histomicstk.cli.utils import CLIArgumentParser
from glob import glob

import sys
sys.path.append("..")
from deeplab.utils.mask_to_xml import xml_create, xml_add_annotation, xml_add_region, xml_save
from deeplab.utils.xml_to_mask import write_minmax_to_xml

def main(args):

    host = args.girderApiUrl
    apiKey = args.girderApiKey

    # get compartments
    compartments = args.classes

    _ = os.system("printf '\n\n---\n\nUsing annotated layers: {}'".format(compartments))
    xml_color=[65280]*len(compartments) # for conversion to xml

    # get folder
    folder = args.inputFolder
    girder_folder_id = folder.split('/')[-2]
    _ = os.system("printf 'Using data from girder_client Folder: {}\n'".format(folder))

    os.system("ls -lh '{}'".format(folder))

    # folder = '5fa18ecdf653e3ea051a2766'
    # init_model = 'model.ckpt-400000'
    patch_size = args.patch_size
    batch_size = args.batch_size
    steps = args.steps

    # create tmp directory for storing intemediate files
    if not args.use_xml:
        tmp = '{}/network_training_data/'.format(folder)
        os.mkdir(tmp)
    else:
        tmp = folder

    def get_base_model_name(model_file):
        try:
            base_model = model_file.split('.meta')
            assert len(base_model) == 2
            base_model = base_model[0]
        except:
            try:
                base_model = model_file.split('.index')
                assert len(base_model) == 2
                base_model = base_model[0]
            except:
                try:
                    base_model = model_file.split('.data')
                    assert len(base_model) == 2
                    base_model = base_model[0]
                except:
                    base_model = model_file
        return base_model

    cwd = os.getcwd()
    # move to data folder and extract models
    os.chdir(tmp)
    # unpck model files from zipped folder
    with open(args.inputModelFile, 'rb') as fh:
        z = zipfile.ZipFile(fh)
        for name in z.namelist():
            z.extract(name, tmp)
    # get num_classes from json file
    with open('args.txt', 'rb') as file:
        trainingDict = json.load(file)
    num_classes = trainingDict['num_classes']

    # move back to cli folder
    os.chdir(cwd)

    model_files = glob('{}/*.ckpt*'.format(tmp))
    print(model_files)
    model_file = model_files[0]
    init_model = get_base_model_name(model_file)

    if args.use_xml:
        slides_used = glob('{}/*.xml'.format(tmp))

    if not args.use_xml:
        gc = girder_client.GirderClient(apiUrl='{}'.format(host))
        gc.authenticate(apiKey=apiKey)
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
            _ = os.system("printf '\tFETCHING SLIDE...\n'")
            os.rename('{}/{}'.format(folder, slidename), '{}/{}'.format(tmp, slidename))
            slides_used.append(slidename)
            #gc.downloadItem(file['_id'], tmp)

            # save the final xml file
            xml_path = '{}/{}.xml'.format(tmp, os.path.splitext(slidename)[0])
            _ = os.system("printf '\tsaving a created xml annotation file: [{}]\n'".format(xml_path))
            xml_save(Annotations=xmlAnnot, filename=xml_path)
            write_minmax_to_xml(xml_path) # to avoid trying to write to the xml from multiple workers
            del xmlAnnot

    os.system("ls -lh '{}'".format(tmp))
    _ = os.system("printf '\ndone retriving data...\nstarting training...\n\n'")

    # setup training params cli args
    trainlogdir = '{}/traininglogs/'.format(tmp)
    augment = args.augment
    scales = args.WSI_downsample
    batch_norm = args.batch_norm
    base_learning_rate = args.learning_rate
    start_learn_rate = args.learning_rate_start
    slow_start_step = args.slow_start_step
    init_last_layer = args.init_last_layer

    # add training metadata to training zip file
    comparts = ','.join(compartments)
    trainingDict = {'num_classes':len(compartments)+1, 'compartments':comparts, 'patch_size':patch_size, 'batch_size':batch_size, 'steps':steps, 'init_model':os.path.basename(args.inputModelFile), 'slides_used':slides_used}
    os.mkdir(trainlogdir)
    with open('{}/args.txt'.format(trainlogdir), 'w') as file:
        file.write(json.dumps(trainingDict))

    cmd = "python3 ../deeplab/train.py --model_variant xception_65 --atrous_rates 6 --atrous_rates 12 --atrous_rates 18 --output_stride 16 --decoder_output_stride 4 --train_crop_size '{}' --train_logdir {} --dataset_dir {} --logtostderr --train_batch_size '{}' --num_clones 1 --tf_initial_checkpoint {} --training_number_of_steps '{}' --learning_rate_decay_step '{}' --slow_start_step {} --augment_prob {} --slow_start_learning_rate {} --base_learning_rate {} --train_model_zipfile {} --save_interval_secs 600 --num_clones {}".format(patch_size, trainlogdir.replace(' ', '\ '), tmp.replace(' ', '\ '), batch_size, init_model.replace(' ', '\ '), steps, round(steps/20), slow_start_step, augment, start_learn_rate, base_learning_rate, args.output_model, args.num_clones)

    for scale in scales:
        cmd += ' --wsi_downsample {}'.format(scale)

    if not init_last_layer:
        cmd += ' --initialize_last_layer=false'

    if not batch_norm:
        cmd += ' --fine_tune_batch_norm=false'

    # run training
    os.system("printf '{}\n'".format(cmd))
    os.system(cmd)

    # move model to zipped file for output
    os.listdir(trainlogdir)
    os.chdir(trainlogdir)
    os.system('pwd')
    os.system('ls -lh')

    # get newest model
    filelist = glob('*.ckpt*')
    latest_model = max(filelist, key=os.path.getmtime)
    # get all ckpt files for latest model
    base_model_name = get_base_model_name(latest_model)
    models = glob('{}*'.format(base_model_name))
    # zip models into new folder
    z = zipfile.ZipFile(args.output_model, 'w')
    for model in models:
        z.write(model, compress_type=zipfile.ZIP_DEFLATED)
    z.write('args.txt', compress_type=zipfile.ZIP_DEFLATED)
    z.close()


if __name__ == "__main__":
    main(CLIArgumentParser().parse_args())
