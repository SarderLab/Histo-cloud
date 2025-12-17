### first

updated the following files to format them in TF2 idiomatic APIs. Looks into TF_MIGRATION_PLAN.md to see what changes I made in those files.

1. core/utils.py
2. core/preprocess_utils.py
3. common.py
4. core/xception.py
5. core/resnet_v1_beta.py
6. nets/mobilenet/\*.py
7. core/feature_extractor.py
8. model.py
9. datasets/data_generator.py
10. datasets/wsi_data_generator.py
11. input_preprocess.py
12. created new file `train_tf2.py` replacing `train.py`
13. created new file `eval_tf2.py` replacing `eval.py`
14. created new file `export_model_tf2.py` replacing `export_model.py`
15. created `tf1_to_tf2_mapper.py` which I will use to map weights
