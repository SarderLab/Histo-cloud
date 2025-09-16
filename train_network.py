import sys
import girder_client, os, json, zipfile, logging
import argparse
import numpy as np

from glob import glob
from deeplab.utils.mask_to_xml import xml_create, xml_add_annotation, xml_add_region, xml_save
from deeplab.utils.xml_to_mask import write_minmax_to_xml

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', filename='logs/train_network.log', filemode='w')

def create_xmls(args, girder_folder_id, folder, tmp, compartments, ignore_label):
    # get girder client
    gc = girder_client.GirderClient(apiUrl=args.girderApiUrl)
    gc.setToken(args.girderToken)

    logging.info('\n\nProcessing data for annotation layers...\n\n')

    # for conversion to xml
    xml_color=[65280]*(len(compartments)+1)

    # create files folder
    save_dir = "files"
    os.makedirs(save_dir, exist_ok=True)

    # get files in folder
    files = list(gc.listItem(girder_folder_id))

    xml_annots = list([file['name'] for file in files.copy() if file['name'].split('.')[1] == 'xml'])
    logging.info(xml_annots)
    # download slides and annotations to tmp directory
    slides_used = []
    for file in files:
        slide = file['name']
        slidename = slide.split('.')[0]
        ext = slide.split('.')[1]

        if ext != 'svs':
            continue

        logging.info('\n---\n\nFOUND: [{}]\n'.format(slide))
        skipSlide = 0

        if '{}.xml'.format(slidename) in xml_annots:
            # Expectation is that the xml file is in the same folder as the slide and has the same name
            logging.info('\n\tFOUND XML Annotation file: [{}.xml]\n'.format(slidename))
            xml_id = [file['_id'] for file in files.copy() if file['name'] == '{}.xml'.format(slidename)][0]
            #Download the xml file
            gc.downloadItem(xml_id, save_dir)
            xml_annots.remove('{}.xml'.format(slidename))
            xml_path = '{}/{}'.format(save_dir, '{}.xml'.format(slidename))
        else:
            # If no xml file is found, create one
            logging.info('\n\tNO XML Annotation file found: [{}]\n'.format(slide))
            # get annotation
            item = gc.getItem(file['_id'])
            annot = gc.get('/annotation/item/{}'.format(item['_id']), parameters={'sort': 'updated'})
            annot.reverse()
            annot = list(annot)
            logging.info("\tfound [{}] annotation layers...\n".format(len(annot)))

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
                        logging.info('\tloading annotation layer: [{}]\n'.format(compart))

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

                        break

            if skipSlide != len(compartments):
                logging.info('\tThis slide is missing annotation layers\n')
                logging.info('\tSKIPPING SLIDE...\n')
                del xmlAnnot
                # correct layers not present
                continue

            # add ignore label if present
            compart = args.ignore_label

            # add layer to xml
            xmlAnnot = xml_add_annotation(Annotations=xmlAnnot, xml_color=xml_color, annotationID=ignore_label)
            
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
                    logging.info('\tloading annotation layer: [{}]\n'.format(compart))
                    a_data = a['annotation']['elements']
                    for data in a_data:
                        pointList = []
                        if data['type'] == 'polyline':
                            points = data['points']
                        elif data['type'] == 'rectangle':
                            center = data['center']
                            width = data['width']/2
                            height = data['height']/2
                            points = [[ center[0]-width, center[1]-width ],[ center[0]+width, center[1]+width ]]
                        for point in points:
                            pt_dict = {'X': round(point[0]), 'Y': round(point[1])}
                            pointList.append(pt_dict)
                        pointsList.append(pointList)
                    # write annotations to xml
                    for i in range(np.shape(pointsList)[0]):
                        pointList = pointsList[i]
                        xmlAnnot = xml_add_region(Annotations=xmlAnnot, pointList=pointList, annotationID=ignore_label)
                    break

            # include slide and fetch annotations
            logging.info('\tFETCHING SLIDE...\n')
            os.rename('{}/{}'.format(folder, slide), '{}/{}'.format(tmp, slide))
            slides_used.append(slide)
            #gc.downloadItem(file['_id'], tmp)

            # save the final xml file
            xml_path = '{}/{}/{}.xml'.format(tmp, file['name'], os.path.splitext(slide)[0])
            logging.info('\tsaving a created xml annotation file: [{}]\n'.format(xml_path))
            xml_save(Annotations=xmlAnnot, filename=xml_path)
            # to avoid trying to write to the xml from multiple workers
            write_minmax_to_xml(xml_path)
            # upload xml to girder
            del xmlAnnot

        # Upload the XML file under respective item as file
        gc.uploadFileToItem(file['_id'], xml_path, 'annotations')

    logging.info('\ndone retriving data...\n')
    return slides_used


def main(args):
    logging.info(args)

    # get compartments
    compartments = args.classes

    logging.info("\n\n---\n\nUsing annotated layers: {}\n\n".format(compartments))

    # get folder
    folder = args.inputFolder
    logging.info(os.listdir(folder))

    girder_folder_id = args.inputFolderID
    logging.info('\nUsing data from girder_client Folder: {}\n'.format(folder))
    
    os.system("ls -lh '{}'".format(folder))

    patch_size = args.patch_size
    batch_size = args.batch_size
    steps = args.steps

    # create tmp directory for storing intemediate files
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
    
    logging.info(os.system('ls -lh'))

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
    logging.info(model_files)
    model_file = model_files[0]
    init_model = get_base_model_name(model_file)

    ignore_label = len(compartments)+1
    slides_used = create_xmls(args, girder_folder_id, folder, tmp, compartments, ignore_label)

    
    logging.info('starting training...\n\n')

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
    trainingDict = {
        'num_classes':len(compartments)+1, 
        'compartments':comparts, 
        'patch_size':patch_size, 
        'batch_size':batch_size, 
        'steps':steps, 
        'init_model':os.path.basename(args.inputModelFile), 
        'slides_used':slides_used
    }

    if not os.path.exists(trainlogdir):
        os.mkdir(trainlogdir)

    with open('{}/args.txt'.format(trainlogdir), 'w') as file:
        file.write(json.dumps(trainingDict))


    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]="{}".format(args.gpu)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    cmd = "python3 ./deeplab/train.py --model_variant xception_65 --atrous_rates 6 --atrous_rates 12 --atrous_rates 18 --output_stride 16 --decoder_output_stride 4 --train_crop_size '{}' --train_logdir {} --dataset_dir {} --logtostderr --train_batch_size '{}' --tf_initial_checkpoint {} --training_number_of_steps '{}' --slow_start_step {} --augment_prob {} --slow_start_learning_rate {} --base_learning_rate {} --train_model_zipfile {} --save_interval_secs 600 --num_clones {} --global_step {} --end_learning_rate {} --learning_power {} --ignore_label {} --decay_steps {} --last_layer_gradient_multiplier {}".format(patch_size, trainlogdir.replace(' ', '\ '), tmp.replace(' ', '\ '), batch_size, init_model.replace(' ', '\ '), steps, slow_start_step, augment, start_learn_rate, base_learning_rate, args.outputModel.replace(' ', '\ '), args.num_clones, args.global_step, args.end_learning_rate, args.learning_power, ignore_label, args.decay_steps, args.last_layer_gradient_multiplier)

    for scale in scales:
        cmd += ' --wsi_downsample {}'.format(scale)

    if not init_last_layer:
        cmd += ' --initialize_last_layer=false'
    else:
        cmd += ' --initialize_last_layer=true'

    if not batch_norm:
        cmd += ' --fine_tune_batch_norm=false'
    else:
        cmd += ' --fine_tune_batch_norm=true'

    if not args.last_layers_contain_logits_only:
        cmd += ' --last_layers_contain_logits_only=false'
    else:
        cmd += ' --last_layers_contain_logits_only=true'

    if not args.upsample_logits:
        cmd += ' --upsample_logits=false'
    else:
        cmd += ' --upsample_logits=true'

    # run training
    logging.info('{}\n'.format(cmd))

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
    z = zipfile.ZipFile(args.outputModel, 'w')
    for model in models:
        z.write(model, compress_type=zipfile.ZIP_DEFLATED)
    z.write('args.txt', compress_type=zipfile.ZIP_DEFLATED)
    z.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a network on a set of annotated slides')
    parser.add_argument('--girderApiUrl', type=str, help='Girder API URL')
    parser.add_argument('--girderToken', type=str, help='Girder API Token')
    parser.add_argument('--inputFolder', type=str, help='Girder Folder ID containing slides and annotations')
    parser.add_argument('--inputFolderID', type=str, help='Girder Folder ID containing slides and annotations')
    parser.add_argument('--inputModelFile', type=str, help='Girder File ID containing the model')
    parser.add_argument('--outputModel', type=str, help='Output model file')
    parser.add_argument('--classes', nargs='+', help='List of classes to train on')
    parser.add_argument('--patch_size', type=int, help='Patch size')
    parser.add_argument('--batch_size', type=int, help='Batch size')
    parser.add_argument('--steps', type=int, help='Number of steps')
    parser.add_argument('--WSI_downsample', nargs='+', type=int, help='List of downsample factors for WSI')
    parser.add_argument('--learning_rate', type=float, help='Learning rate')
    parser.add_argument('--learning_rate_start', type=float, help='Learning rate start')
    parser.add_argument('--slow_start_step', type=int, help='Slow start step')
    parser.add_argument('--init_last_layer', type=bool, help='Initialize last layer')
    parser.add_argument('--batch_norm', type=bool, help='Batch normalization')
    parser.add_argument('--augment', type=float, help='Augmentation probability')
    parser.add_argument('--num_clones', type=int, help='Number of clones')
    parser.add_argument('--global_step', type=int, help='Global step')
    parser.add_argument('--end_learning_rate', type=float, help='End learning rate')
    parser.add_argument('--learning_power', type=float, help='Learning power')
    parser.add_argument('--ignore_label', type=str, help='Ignore label')
    parser.add_argument('--decay_steps', type=int, help='Decay steps')
    parser.add_argument('--last_layer_gradient_multiplier', type=int, help='Last layer gradient multiplier')
    parser.add_argument('--last_layers_contain_logits_only', type=bool, help='Last layers contain logits only')
    parser.add_argument('--upsample_logits', type=bool, help='Upsample logits')
    parser.add_argument('--gpu', type=int, help='GPU')
    args = parser.parse_args()

    main(args)
