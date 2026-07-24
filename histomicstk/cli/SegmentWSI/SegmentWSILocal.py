import os, zipfile, json
import argparse
from glob import glob
import shutil
import subprocess
import sys
import tempfile

def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ('true', '1', 'yes', 'y', 'on'):
        return True
    if value.lower() in ('false', '0', 'no', 'n', 'off'):
        return False
    raise argparse.ArgumentTypeError('Expected a boolean value.')

def build_arg_parser():
    """Define the local equivalent of the parameters in SegmentWSI.xml."""
    parser = argparse.ArgumentParser(
        description='Segments structures from a whole-slide image.'
    )

    # IO parameters: SegmentWSI.xml lines 14-34.
    parser.add_argument('--inputImageFile', required=True,
        help='Path to the input whole-slide image.')
    parser.add_argument('--inputModelFile', required=True,
        help='Path to the input TensorFlow model zip file.')
    parser.add_argument('--outputAnnotationFile', required=True,
        help='Path for the output annotation (*.anot) file.')

    # Heatmap parameters: SegmentWSI.xml lines 39-52.
    parser.add_argument('--save_heatmap', type=str_to_bool, default=False,  
        help='Save network logits as heatmap layers (true/false).')
    parser.add_argument('--heatmap_stride', type=int, default=2,
        help='Additional downsample applied to saved heatmaps.')

    # WSI analysis parameters: SegmentWSI.xml lines 57-112.
    parser.add_argument('--wsi_downsample', type=int, default=2)
    parser.add_argument('--patch_size', type=int, default=2000)
    parser.add_argument('--tile_stride', type=int, default=1000)
    parser.add_argument('--remove_border', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--min_size', type=int, default=2000)
    parser.add_argument('--simplify_contours', type=float, default=0.005)
    parser.add_argument('--gpu', type=int, default=0)
    return parser

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

    output_dir = os.path.dirname(os.path.abspath(args.outputAnnotationFile))
    os.makedirs(output_dir, exist_ok=True)
    print(output_dir)

    # Extract into an isolated directory so checkpoints from previous runs in
    # the annotation output directory cannot be selected.
    tmp = tempfile.mkdtemp(prefix='segment_wsi_model_', dir=output_dir)
    try:
        with zipfile.ZipFile(args.inputModelFile) as z:
            z.extractall(tmp)
        with open(os.path.join(tmp, 'args.txt'), 'rb') as file:
            trainingDict = json.load(file)
        num_classes = trainingDict['num_classes']
        compartments = trainingDict['compartments']

        model_files = glob('{}/**/*.ckpt*'.format(tmp), recursive=True)
        if not model_files:
            raise FileNotFoundError('No *.ckpt* files found in model zip.')

        print(model_files)
        model = get_base_model_name(model_files[0])

        print('\noutput filename: {}\n'.format(args.outputAnnotationFile))

        # Unlike the CTK container entrypoint, a local caller may start in any
        # directory.  Resolve vis.py from this script's location.
        cli_dir = os.path.dirname(os.path.abspath(__file__))
        vis_py = os.path.abspath(os.path.join(cli_dir, '..', '..', 'deeplab', 'vis.py'))
        cmd = "{} {} --model_variant xception_65 --atrous_rates 6 --atrous_rates 12 --atrous_rates 18 --output_stride 16 --decoder_output_stride 4 --save_json_annotation True --checkpoint_dir {} --dataset_dir '{}' --json_filename '{}' --vis_crop_size {} --wsi_downsample {} --tile_step {} --min_size {} --vis_batch_size {} --vis_remove_border {} --simplify_contours {} --num_classes {} --class_names '{}' --save_heatmap {} --heatmap_stride {} --gpu {}".format(
            sys.executable, vis_py, model, args.inputImageFile,
            args.outputAnnotationFile, args.patch_size, args.wsi_downsample,
            args.tile_stride, args.min_size, args.batch_size, args.remove_border,
            args.simplify_contours, num_classes, compartments, args.save_heatmap,
            args.heatmap_stride, args.gpu)
        print(cmd)
        sys.stdout.flush()
        subprocess.check_call(cmd, shell=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

if __name__ == '__main__':
    main(build_arg_parser().parse_args())