# DeepLab-WSI: Deep Labelling for WSI Semantic Image Segmentation

DeepLab is a state-of-art deep learning model for semantic image segmentation,
This code has been modified to work natively and efficiently on Whole Slide Images (WSIs).
This code efficiently pulls randomly selected WSI ROI (patches) at runtime to avoid the need for WSI chopping.

If you find the code useful for your research, please consider citing our latest
works:

*   Cloud based WSI segmentation (arXiv deposition):

```
@misc{lutnick2021tool,
      title={A tool for user friendly, cloud based, whole slide image segmentation},
      author={Brendon Lutnick and David Manthey and Pinaki Sarder},
      year={2021},
      eprint={2101.07222},
      archivePrefix={arXiv},
      primaryClass={eess.IV}
}
```

*   Histofetch - On the fly patch processing:

```
@misc{lutnick2021histofetch,
      title={Histo-fetch -- On-the-fly processing of gigapixel whole slide images simplifies and speeds neural network training},
      author={Brendon Lutnick and Leema Krishna Murali and Brandon Ginley and Avi Z. Rosenberg and Pinaki Sarder},
      year={2021},
      eprint={2102.11433},
      archivePrefix={arXiv},
      primaryClass={eess.IV}
}
```

The DeepLab implementation supports adopting the following network
backbones:

1.  MobileNetv2 [8] and MobileNetv3 [16]: A fast network structure designed
    for mobile devices.

2.  Xception [9, 10]: A powerful network structure intended for server-side
    deployment.

3.  ResNet-v1-{50,101} [14]: We provide both the original ResNet-v1 and its
    'beta' variant where the 'stem' is modified for semantic segmentation.

4.  PNASNet [15]: A Powerful network structure found by neural architecture
    search.

5.  Auto-DeepLab (called HNASNet in the code): A segmentation-specific network
    backbone found by neural architecture search.

For WSI segmentation all tests were done using the Xception-65 backbone, with atrous rates 6, 12, and 18, output_stride=16, and decoder_output_stride=4

## Suggested Usage Examples

### Annotation:
This code uses WSI contour annotations in XML form. This format is the same that is used by [Aperio Imagescope](https://www.leicabiosystems.com/digital-pathology/manage/aperio-imagescope/). If you are using the code standalone (not as a plugin in HistomicsTK-deeplab) we suggest annotating in Imagescope.

In the XML file, annotation layers are defined using the AnnotationID tag. In Imagescope this means that they are built up sequentially as annotation layers.

We also have useful codes for conversion between rasterized masks and XML contour annotations and JSON annotation (for HistomicsUI):
*   [XML --> mask](https://github.com/SarderLab/HistomicsTK-deeplab/blob/main/histomicstk/deeplab/utils/xml_to_mask.py)
*   [mask --> XML](https://github.com/SarderLab/HistomicsTK-deeplab/blob/main/histomicstk/deeplab/utils/mask_to_xml.py)
*   [XML --> JSON](https://github.com/SarderLab/HistomicsTK-deeplab/blob/main/histomicstk/deeplab/utils/xml_to_json.py)

### Training:
python3 [train.py](https://github.com/SarderLab/HistomicsTK-deeplab/blob/main/histomicstk/deeplab/train.py) --model_variant xception_65 --atrous_rates 6 --atrous_rates 12 --atrous_rates 18 --output_stride 16 --decoder_output_stride 4 --train_crop_size 512 --train_logdir <directory to save models> --tf_initial_checkpoint <pretrained model> --dataset_dir <directory with training data> --fine_tune_batch_norm False --train_batch_size 12 --training_number_of_steps 10000 --slow_start_step 1000 --wsi_downsample 1 --wsi_downsample 2 --wsi_downsample 3 --wsi_downsample 4 --augment_prob .1 --slow_start_learning_rate .0001 --base_learning_rate 0.0007 --last_layer_gradient_multiplier 10

### Testing:
python3 [vis.py](https://github.com/SarderLab/HistomicsTK-deeplab/blob/main/histomicstk/deeplab/vis.py) --model_variant xception_65 --atrous_rates 6 --atrous_rates 12 --atrous_rates 18 --output_stride 16 --decoder_output_stride 4 --vis_crop_size 2000 --wsi_downsample 2 --tile_step 1000 --min_size 1000 --vis_batch_size 3 --vis_remove_border 100 --num_classes <set to the number of training classes (number of annotation layers + background)> --dataset_dir <folder with WSIs> --checkpoint_dir <path to trained model>

## Contacts (Maintainers)

*   Brendon Lutnick, github: [brendonlutnick](https://github.com/brendonlutnick)

## Pre-trained Models

*   <a href='g3doc/model_zoo.md'>Checkpoints and frozen inference graphs.</a><br>
*   <a href='https://athena.ccr.buffalo.edu/#collection/5fa17ef9e8737fef305946fe'>Models trained on histology tissue.</a><br>

## License

All the codes in deeplab folder is covered by the [LICENSE](https://github.com/tensorflow/models/blob/master/LICENSE)
under tensorflow/models. Please refer to the LICENSE for details.

## References

1.  **Semantic Image Segmentation with Deep Convolutional Nets and Fully Connected CRFs**<br />
    Liang-Chieh Chen+, George Papandreou+, Iasonas Kokkinos, Kevin Murphy, Alan L. Yuille (+ equal
    contribution). <br />
    [[link]](https://arxiv.org/abs/1412.7062). In ICLR, 2015.

2.  **DeepLab: Semantic Image Segmentation with Deep Convolutional Nets,**
    **Atrous Convolution, and Fully Connected CRFs** <br />
    Liang-Chieh Chen+, George Papandreou+, Iasonas Kokkinos, Kevin Murphy, and Alan L Yuille (+ equal
    contribution). <br />
    [[link]](http://arxiv.org/abs/1606.00915). TPAMI 2017.

3.  **Rethinking Atrous Convolution for Semantic Image Segmentation**<br />
    Liang-Chieh Chen, George Papandreou, Florian Schroff, Hartwig Adam.<br />
    [[link]](http://arxiv.org/abs/1706.05587). arXiv: 1706.05587, 2017.

4.  **Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation**<br />
    Liang-Chieh Chen, Yukun Zhu, George Papandreou, Florian Schroff, Hartwig Adam.<br />
    [[link]](https://arxiv.org/abs/1802.02611). In ECCV, 2018.

5.  **ParseNet: Looking Wider to See Better**<br />
    Wei Liu, Andrew Rabinovich, Alexander C Berg<br />
    [[link]](https://arxiv.org/abs/1506.04579). arXiv:1506.04579, 2015.

6.  **Pyramid Scene Parsing Network**<br />
    Hengshuang Zhao, Jianping Shi, Xiaojuan Qi, Xiaogang Wang, Jiaya Jia<br />
    [[link]](https://arxiv.org/abs/1612.01105). In CVPR, 2017.

7.  **Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate shift**<br />
    Sergey Ioffe, Christian Szegedy <br />
    [[link]](https://arxiv.org/abs/1502.03167). In ICML, 2015.

8.  **MobileNetV2: Inverted Residuals and Linear Bottlenecks**<br />
    Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, Liang-Chieh Chen<br />
    [[link]](https://arxiv.org/abs/1801.04381). In CVPR, 2018.

9.  **Xception: Deep Learning with Depthwise Separable Convolutions**<br />
    François Chollet<br />
    [[link]](https://arxiv.org/abs/1610.02357). In CVPR, 2017.

10. **Deformable Convolutional Networks -- COCO Detection and Segmentation Challenge 2017 Entry**<br />
    Haozhi Qi, Zheng Zhang, Bin Xiao, Han Hu, Bowen Cheng, Yichen Wei, Jifeng Dai<br />
    [[link]](http://presentations.cocodataset.org/COCO17-Detect-MSRA.pdf). ICCV COCO Challenge
    Workshop, 2017.

11. **Tensorflow: Large-Scale Machine Learning on Heterogeneous Distributed Systems**<br />
    M. Abadi, A. Agarwal, et al. <br />
    [[link]](https://arxiv.org/abs/1603.04467). arXiv:1603.04467, 2016.

12. **The Pascal Visual Object Classes Challenge – A Retrospective,** <br />
    Mark Everingham, S. M. Ali Eslami, Luc Van Gool, Christopher K. I. Williams, John
    Winn, and Andrew Zisserma. <br />
    [[link]](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/). IJCV, 2014.

13. **The Cityscapes Dataset for Semantic Urban Scene Understanding**<br />
    Cordts, Marius, Mohamed Omran, Sebastian Ramos, Timo Rehfeld, Markus Enzweiler, Rodrigo Benenson, Uwe Franke, Stefan Roth, Bernt Schiele. <br />
    [[link]](https://www.cityscapes-dataset.com/). In CVPR, 2016.

14. **Deep Residual Learning for Image Recognition**<br />
    Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. <br />
    [[link]](https://arxiv.org/abs/1512.03385). In CVPR, 2016.

15. **Progressive Neural Architecture Search**<br />
    Chenxi Liu, Barret Zoph, Maxim Neumann, Jonathon Shlens, Wei Hua, Li-Jia Li, Li Fei-Fei, Alan Yuille, Jonathan Huang, Kevin Murphy. <br />
    [[link]](https://arxiv.org/abs/1712.00559). In ECCV, 2018.

16. **Searching for MobileNetV3**<br />
    Andrew Howard, Mark Sandler, Grace Chu, Liang-Chieh Chen, Bo Chen, Mingxing Tan, Weijun Wang, Yukun Zhu, Ruoming Pang, Vijay Vasudevan, Quoc V. Le, Hartwig Adam. <br />
    [[link]](https://arxiv.org/abs/1905.02244). In ICCV, 2019.
