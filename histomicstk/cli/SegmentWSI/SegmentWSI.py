import os, zipfile, json
from histomicstk.cli.utils import CLIArgumentParser
from glob import glob

def main(args):

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
    print(cwd)

    tmp = args.outputAnnotationFile
    tmp = os.path.dirname(tmp)
    print(tmp)

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
    compartments = trainingDict['compartments']

    # move back to cli folder
    os.chdir(cwd)

    model_files = glob('{}/*.ckpt*'.format(tmp))
    print(model_files)
    model_file = model_files[0]
    model = get_base_model_name(model_file)

    # list files code can see
    os.system('ls -l {}'.format(model.split('model.ckpt')[0]))

    print('\noutput filename: {}\n'.format(args.outputAnnotationFile))

    # run vis.py with flags
    cmd = "python3 ../deeplab/vis.py --model_variant xception_65 --atrous_rates 6 --atrous_rates 12 --atrous_rates 18 --output_stride 16 --decoder_output_stride 4 --save_json_annotation True --checkpoint_dir {} --dataset_dir '{}' --json_filename '{}' --vis_crop_size {} --wsi_downsample {} --tile_step {} --min_size {} --vis_batch_size {} --vis_remove_border {} --simplify_contours {} --num_classes {} --class_names '{}' --save_heatmap={} --heatmap_stride {} --gpu {}".format(model, args.inputImageFile, args.outputAnnotationFile, args.patch_size, args.wsi_downsample, args.tile_stride, args.min_size, args.batch_size, args.remove_border, args.simplify_contours, num_classes, compartments, args.save_heatmap, args.heatmap_stride, args.gpu)
    os.system(cmd)


if __name__ == "__main__":
    main(CLIArgumentParser().parse_args())
