import os
from histomicstk.cli.utils import CLIArgumentParser

def main(args):

    print(os.getcwd())

    model_file = args.inputModelFile
    try:
        model = model_file.split('.meta')
        assert len(model) == 2
        model = model[0]
    except:
        try:
            model = model_file.split('.index')
            assert len(model) == 2
            model = model[0]
        except:
            try:
                model = model_file.split('.data')
                assert len(model) == 2
                model = model[0]
            except:
                model = model_file

    # list files code can see
    os.system('ls -l {}'.format(model.split('model.ckpt')[0]))

    print('\noutput filename: {}\n'.format(args.outputAnnotationFile))

    # run vis.py with flags
    cmd = "python3 ../deeplab/vis.py --model_variant xception_65 --atrous_rates 6 --atrous_rates 12 --atrous_rates 18 --output_stride 16 --decoder_output_stride 4 --save_json_annotation True --checkpoint_dir {} --dataset_dir '{}' --json_filename '{}' --vis_crop_size {} --wsi_downsample {} --overlap_num {} --min_size {} --vis_batch_size {} --vis_remove_border {}".format(model, args.inputImageFile, args.outputAnnotationFile, args.patch_size, args.wsi_downsample, args.patch_overlap, args.min_size, args.batch_size, args.remove_border)
    os.system(cmd)


if __name__ == "__main__":
    main(CLIArgumentParser().parse_args())
