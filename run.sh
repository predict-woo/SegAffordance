python train_SF3D.py --config config/sf3d_train_full.yaml

python datasets/opdreal.py --data-path /cluster/project/cvg/students/andrye/dataset/MotionDataset_h5_real --dataset-key MotionNet_train --output-dir debug

python evaluation/eval_OPDReal.py \
    --config config/opd_train_large_beta.yaml \
    --checkpoint "/cluster/project/cvg/students/andrye/experiments/OPDReal_v6/epoch=11-val/loss_total=0.5839.ckpt" \
    --data_path /cluster/project/cvg/students/andrye/dataset/MotionDataset_h5_real \
    --dataset_key MotionNet_test

python test_OPDReal.py \
    --config config/opd_train_large_beta.yaml \
    --checkpoint "/cluster/project/cvg/students/andrye/experiments/OPDReal_v7/epoch=11-val/loss_total=0.5769.ckpt" \
    --dataset_key MotionNet_test \
    --motion_threshold 10.0 \
    --iou_threshold 0.5

python test_OPDReal.py \
    --config config/opd_train_large_beta.yaml \
    --checkpoint "/cluster/project/cvg/students/andrye/experiments/OPDReal_v7/epoch=11-val/loss_total=0.5769.ckpt" \
    --dataset_key MotionNet_test \
    --motion_threshold 10.0 \
    --iou_threshold 0.5

python train_OPDReal.py \
    --config config/opd_train_large_beta.yaml

conda activate /cluster/project/cvg/students/andrye/miniconda3/envs/segaffordance && python test_OPDReal.py \
    --config config/opd_train_large_beta.yaml \
    --checkpoint "/cluster/project/cvg/students/andrye/experiments/OPDReal_w_motion_type_v1/epoch=2-val/loss_total=1.0314.ckpt" \
    --dataset_key MotionNet_test \
    --motion_threshold 10.0 \
    --iou_threshold 0.5 \
    --pred_threshold 0.3

conda activate /cluster/project/cvg/students/andrye/miniconda3/envs/segaffordance && python test_OPDReal.py \
    --config config/opd_train_large_beta.yaml \
    --checkpoint "/cluster/project/cvg/students/andrye/experiments/OPDReal_w_motion_type_v1/epoch=2-val/loss_total=1.0314.ckpt" \
    --dataset_key MotionNet_test \
    --motion_threshold 10.0 \
    --iou_threshold 0.5 \
    --pred_threshold 0.4

conda activate /cluster/project/cvg/students/andrye/miniconda3/envs/segaffordance && python test_OPDReal.py \
    --config config/opd_train_large_beta.yaml \
    --checkpoint "/cluster/project/cvg/students/andrye/experiments/OPDReal_w_motion_type_v1/epoch=2-val/loss_total=1.0314.ckpt" \
    --dataset_key MotionNet_test \
    --motion_threshold 10.0 \
    --iou_threshold 0.5 \
    --pred_threshold 0.5

conda activate /cluster/project/cvg/students/andrye/miniconda3/envs/segaffordance && python test_OPDReal.py \
    --config config/opd_train_large_beta.yaml \
    --checkpoint "/cluster/project/cvg/students/andrye/experiments/OPDReal_w_motion_type_v1/epoch=2-val/loss_total=1.0314.ckpt" \
    --dataset_key MotionNet_test \
    --motion_threshold 10.0 \
    --iou_threshold 0.5 \
    --pred_threshold 0.6

conda activate /cluster/project/cvg/students/andrye/miniconda3/envs/segaffordance && python test_OPDReal.py \
    --config config/opd_train_large_beta.yaml \
    --checkpoint "/cluster/project/cvg/students/andrye/experiments/OPDReal_w_motion_type_v1/epoch=2-val/loss_total=1.0314.ckpt" \
    --dataset_key MotionNet_test \
    --motion_threshold 10.0 \
    --iou_threshold 0.5 \
    --pred_threshold 0.7

conda activate /cluster/project/cvg/students/andrye/miniconda3/envs/segaffordance && python test_OPDReal.py \
    --config config/opd_train_large_beta.yaml \
    --checkpoint "/cluster/project/cvg/students/andrye/experiments/OPDReal_w_motion_type_v1/epoch=2-val/loss_total=1.0314.ckpt" \
    --dataset_key MotionNet_test \
    --motion_threshold 10.0 \
    --iou_threshold 0.5 \
    --pred_threshold 0.9


conda activate /cluster/project/cvg/students/andrye/miniconda3/envs/segaffordance && python test_OPDReal.py \
    --config config/opd_train_large_beta.yaml \
    --checkpoint "/cluster/project/cvg/students/andrye/experiments/OPDReal_w_motion_type_v1/epoch=18-val/loss_total=0.8848.ckpt" \
    --dataset_key MotionNet_test \
    --motion_threshold 10.0 \
    --iou_threshold 0.5 \
    --pred_threshold 0.5


conda activate /cluster/project/cvg/students/andrye/miniconda3/envs/segaffordance && python test_OPDReal.py \
    --config config/opd_train_large_beta.yaml \
    --checkpoint "/cluster/project/cvg/students/andrye/experiments/OPDReal_w_motion_type_v4/epoch=10-val/loss_total=0.8462.ckpt" \
    --dataset_key MotionNet_test \
    --motion_threshold 10.0 \
    --iou_threshold 0.5 \
    --pred_threshold 0.5



conda activate /cluster/project/cvg/students/andrye/miniconda3/envs/segaffordance && python test_OPDReal.py \
    --config config/opd_train_large_beta.yaml \
    --checkpoint "/cluster/project/cvg/students/andrye/experiments/OPDReal_w_motion_type_v6/epoch=35-val/loss_total=0.9040.ckpt" \
    --dataset_key MotionNet_test \
    --motion_threshold 10.0 \
    --iou_threshold 0.5 \
    --pred_threshold 0.5

conda activate /cluster/project/cvg/students/andrye/miniconda3/envs/segaffordance && python test_OPDReal.py \
    --config config/opd_train_large_beta.yaml \
    --checkpoint "/cluster/project/cvg/students/andrye/experiments/OPDReal_w_motion_type_v6/epoch=26-val/loss_total=0.8590.ckpt" \
    --dataset_key MotionNet_test \
    --motion_threshold 10.0 \
    --iou_threshold 0.5 \
    --pred_threshold 0.5



conda activate /cluster/project/cvg/students/andrye/miniconda3/envs/segaffordance && python train_OPDMulti.py \
    --config config/opdmulti-train.yaml \
    --finetune_from "/cluster/project/cvg/students/andrye/experiments/OPDReal_w_motion_type_v6/epoch=35-val/loss_total=0.9040.ckpt" \
    --train_only_heads




conda activate /cluster/project/cvg/students/andrye/miniconda3/envs/segaffordance && python test_OPDMulti.py \
    --config config/odpmulti-train.yaml \
    --checkpoint "/cluster/project/cvg/students/andrye/experiments/OPDMulti_v1/epoch=26-val/loss_total=1.2169.ckpt" \
    --dataset_key MotionNet_test \
    --motion_threshold 10.0 \
    --iou_threshold 0.5 \
    --pred_threshold 0.5


conda activate /cluster/project/cvg/students/andrye/miniconda3/envs/segaffordance && python train_OPDMulti.py \
    --config config/opdmulti-train.yaml \
    --finetune_from "/cluster/project/cvg/students/andrye/experiments/OPDReal_w_motion_type_v6/epoch=9-val/loss_total=0.8648.ckpt" \
    --train_only_heads

conda activate /cluster/project/cvg/students/andrye/miniconda3/envs/segaffordance && python3 datasets/filter_bad_annotations.py \
    /cluster/project/cvg/students/andrye/OPDMulti/MotionDataset_h5/annotations_wd/MotionNet_train.json \
    /cluster/project/cvg/students/andrye/OPDMulti/MotionDataset_h5/annotations_wdf/MotionNet_train.json \
    --data_dir /cluster/project/cvg/students/andrye/OPDMulti/MotionDataset_h5 \
    --debug_viz_dir dv

conda activate /cluster/project/cvg/students/andrye/miniconda3/envs/segaffordance && python3 datasets/filter_bad_annotations.py \
    /cluster/project/cvg/students/andrye/OPDMulti/MotionDataset_h5/annotations_wd/MotionNet_valid.json \
    /cluster/project/cvg/students/andrye/OPDMulti/MotionDataset_h5/annotations_wdf/MotionNet_valid.json \
    --data_dir /cluster/project/cvg/students/andrye/OPDMulti/MotionDataset_h5 \
    --is_multi

conda activate /cluster/project/cvg/students/andrye/miniconda3/envs/segaffordance && python3 datasets/filter_bad_annotations.py \
    /cluster/project/cvg/students/andrye/OPDMulti/MotionDataset_h5/annotations_wd/MotionNet_test.json \
    /cluster/project/cvg/students/andrye/OPDMulti/MotionDataset_h5/annotations_wdf/MotionNet_test.json \
    --data_dir /cluster/project/cvg/students/andrye/OPDMulti/MotionDataset_h5 \
    --is_multi


python train_OPDReal.py \
    --config config/opd_train_large_beta.yaml

conda activate /cluster/project/cvg/students/andrye/miniconda3/envs/segaffordance && python train_OPDMulti.py \
    --config config/opdmulti-train.yaml \
    --finetune_from "/cluster/project/cvg/students/andrye/experiments/OPDReal_w_motion_type_v8/epoch=17-val/loss_total=0.6532.ckpt" \
    --train_only_heads



conda activate /cluster/project/cvg/students/andrye/miniconda3/envs/segaffordance && python test_OPDMulti.py \
    --config config/opdmulti-train.yaml \
    --checkpoint "/cluster/project/cvg/students/andrye/experiments/OPDMulti_v4/epoch=9-val/loss_total=0.7175.ckpt" \
    --dataset_key MotionNet_test \
    --motion_threshold 10.0 \
    --iou_threshold 0.5 \
    --pred_threshold 0.5


conda activate /cluster/project/cvg/students/andrye/miniconda3/envs/segaffordance && python test_OPDMulti.py \
    --config config/opdmulti-train.yaml \
    --checkpoint "/cluster/project/cvg/students/andrye/experiments/OPDMulti_v4/epoch=6-val/loss_total=0.6859.ckpt" \
    --dataset_key MotionNet_test \
    --motion_threshold 10.0 \
    --iou_threshold 0.5 \
    --pred_threshold 0.5



conda activate /cluster/project/cvg/students/andrye/miniconda3/envs/segaffordance && python3 datasets/filter_bad_annotations.py \
    /cluster/project/cvg/students/andrye/dataset/MotionDataset_h5_real/annotations_bwd/MotionNet_train.json \
    /cluster/project/cvg/students/andrye/dataset/MotionDataset_h5_real/annotations_bwdf/MotionNet_train.json \
    --data_dir /cluster/project/cvg/students/andrye/dataset/MotionDataset_h5_real

conda activate /cluster/project/cvg/students/andrye/miniconda3/envs/segaffordance && python3 datasets/filter_bad_annotations.py \
    /cluster/project/cvg/students/andrye/dataset/MotionDataset_h5_real/annotations_bwd/MotionNet_valid.json \
    /cluster/project/cvg/students/andrye/dataset/MotionDataset_h5_real/annotations_bwdf/MotionNet_valid.json \
    --data_dir /cluster/project/cvg/students/andrye/dataset/MotionDataset_h5_real


python train_OPDReal.py \
    --config config/opd-train.yaml

conda activate /cluster/project/cvg/students/andrye/miniconda3/envs/segaffordance && python test_OPDReal_better.py \
    --config config/opd_train_large_beta.yaml \
    --checkpoint "/cluster/project/cvg/students/andrye/experiments/OPDReal_w_motion_type_v9/epoch=27-val/loss_total=0.5688.ckpt" \
    --dataset_key MotionNet_test \
    --motion_threshold 10.0 \
    --iou_threshold 0.5


conda activate /cluster/project/cvg/students/andrye/miniconda3/envs/segaffordance && python test_OPDMulti.py \
    --config config/opdmulti-train.yaml \
    --checkpoint "/cluster/project/cvg/students/andrye/experiments/OPDReal_w_motion_type_v9/epoch=27-val/loss_total=0.5688.ckpt" \
    --dataset_key MotionNet_test \
    --motion_threshold 10.0 \
    --iou_threshold 0.5




conda activate /cluster/project/cvg/students/andrye/miniconda3/envs/segaffordance && python3 datasets/filter_bad_annotations.py \
    /cluster/project/cvg/students/andrye/OPDMulti/MotionDataset_h5/annotations_bwd/MotionNet_train.json \
    /cluster/project/cvg/students/andrye/OPDMulti/MotionDataset_h5/annotations_bwdf/MotionNet_train.json \
    --data_dir /cluster/project/cvg/students/andrye/OPDMulti/MotionDataset_h5 \
    --debug_viz_dir dv \
    --is_multi

conda activate /cluster/project/cvg/students/andrye/miniconda3/envs/segaffordance && python3 datasets/filter_bad_annotations.py \
    /cluster/project/cvg/students/andrye/OPDMulti/MotionDataset_h5/annotations_bwd/MotionNet_train.json \
    /cluster/project/cvg/students/andrye/OPDMulti/MotionDataset_h5/annotations_bwdf/MotionNet_train.json \
    --data_dir /cluster/project/cvg/students/andrye/OPDMulti/MotionDataset_h5 \
    --is_multi

conda activate /cluster/project/cvg/students/andrye/miniconda3/envs/segaffordance && python3 datasets/filter_bad_annotations.py \
    /cluster/project/cvg/students/andrye/OPDMulti/MotionDataset_h5/annotations_bwd/MotionNet_valid.json \
    /cluster/project/cvg/students/andrye/OPDMulti/MotionDataset_h5/annotations_bwdf/MotionNet_valid.json \
    --data_dir /cluster/project/cvg/students/andrye/OPDMulti/MotionDataset_h5 \
    --is_multi

conda activate /cluster/project/cvg/students/andrye/miniconda3/envs/segaffordance && python train_OPDMulti.py \
    --config config/opdmulti-train.yaml \
    --finetune_from "/cluster/project/cvg/students/andrye/experiments/OPDReal_w_motion_type_v9/epoch=27-val/loss_total=0.5688.ckpt" \
    --train_only_heads



/cluster/project/cvg/students/andrye/experiments/OPDMulti_v5/epoch=11-val/loss_total=0.7197.ckpt

├── epoch=11-val
│   └── loss_total=0.7197.ckpt

conda activate /cluster/project/cvg/students/andrye/miniconda3/envs/segaffordance && python test_OPDMulti_better.py \
    --config /cluster/home/andrye/SegAffordance/config/opdmulti-train.yaml \
    --checkpoint "/cluster/project/cvg/students/andrye/experiments/OPDMulti_v6/epoch=6-val/loss_total=0.6839.ckpt" \
    --dataset_key MotionNet_test \
    --motion_threshold 10.0 \
    --iou_threshold 0.5 \
    --pred_threshold 0.5 \
    --visualize_debug

python datasets/visualizations.py \
    --dataset-name opdreal \
    --data-path /cluster/project/cvg/students/andrye/dataset/MotionDataset_h5_real \
    --num-samples 10 \
    --output-dir debug


conda activate /cluster/project/cvg/students/andrye/miniconda3/envs/segaffordance && python3 datasets/filter_bad_annotations.py \
    /cluster/project/cvg/students/andrye/OPDMulti/MotionDataset_h5/annotations_bwd/MotionNet_test.json \
    /cluster/project/cvg/students/andrye/OPDMulti/MotionDataset_h5/annotations_bwdf/MotionNet_test.json \
    --data_dir /cluster/project/cvg/students/andrye/OPDMulti/MotionDataset_h5 \
    --is_multi



python train_OPDReal_better.py fit --config config/opd_train.yaml --config config/opdreal_train.yaml

python train_OPDReal_better.py fit --config config/opd_train.yaml --config config/opdsynth_train.yaml


python train_OPDMulti_better.py fit --config config/opd_train.yaml --config config/opdmulti_train.yaml


python train_OPDMulti_better.py test \
  --config config/opd_train.yaml \
  --config config/opdmulti_train.yaml \
  --ckpt_path /cluster/project/cvg/students/andrye/experiments/OPDReal_v12_bce/best-epoch=9-val/loss_total=0.5307.ckpt


python train_OPDMulti_better.py test \
  --config config/opd_train.yaml \
  --config config/opdmulti_train.yaml \
  --config config/opdmulti_test.yaml \
  --ckpt_path /cluster/project/cvg/students/andrye/experiments/OPDMulti_v12/best-epoch=10-val/loss_total=0.7878.ckpt


python train_OPDReal_better.py test \
  --config config/opd_train.yaml \
  --config config/opdreal_train.yaml \
  --config config/opdreal_test.yaml \
  --ckpt_path /cluster/project/cvg/students/andrye/experiments/OPDReal_v17/best-epoch=27-val/loss_total=0.5939.ckpt

config/opdmulti_train_no_pretrain.yaml
config/opdmulti_train_no_freeze.yaml


python train_OPDMulti_better.py fit --config config/opd_train.yaml --config config/opdmulti_train_no_pretrain.yaml && python train_OPDMulti_better.py fit --config config/opd_train.yaml --config config/opdmulti_train_no_freeze.yaml



python train_OPDReal_better.py fit --config config/opd_train.yaml --config config/opdreal_train_no_depth.yaml

python train_OPDReal_better.py fit --config config/opd_train.yaml --config config/opdreal_train_no_cvae.yaml


python test_OPDReal_better.py test --config config/opd_train.yaml --config config/opdreal_train.yaml --config config/opdreal_test.yaml --ckpt_path /cluster/project/cvg/students/andrye/experiments/OPDReal_v17/best-epoch=27-val/loss_total=0.5939.ckpt

python test_OPDReal_better.py test \
    --config config/opd_train.yaml \
    --config config/opdmulti_train_no_pretrain.yaml \
    --config config/opdreal_test.yaml \
    --ckpt_path /cluster/project/cvg/students/andrye/experiments/OPDMulti_v14_no_pretrain/best-epoch=27-val/loss_total=0.9321.ckpt


python test_OPDReal_better.py test \
    --config config/opd_train.yaml \
    --config config/opdreal_train_no_depth.yaml \
    --config config/opdreal_test.yaml \
    --ckpt_path /cluster/project/cvg/students/andrye/experiments/OPDReal_v18_no_depth/best-epoch=25-val/loss_total=0.6191.ckpt


python test_OPDReal_better.py test \
    --config config/opd_train.yaml \
    --config config/opdreal_train_no_cvae.yaml \
    --config config/opdreal_test.yaml \
    --ckpt_path /cluster/project/cvg/students/andrye/experiments/OPDReal_v17/best-epoch=25-val/loss_total=0.5861.ckpt

python test_OPDReal_better.py test \
    --config config/opd_train.yaml \
    --config config/opdmulti_train.yaml \
    --config config/opdmulti_test.yaml \
    --ckpt_path /cluster/project/cvg/students/andrye/experiments/OPDMulti_v12/best-epoch=11-val/loss_total=0.7877.ckpt

python test_OPDMulti_better.py test \
    --config config/opd_train.yaml \
    --config config/opdmulti_train_no_freeze.yaml \
    --config config/opdmulti_test.yaml \
    --ckpt_path /cluster/project/cvg/students/andrye/experiments/OPDMulti_v13_no_freeze/best-epoch=4-val/loss_total=0.7669.ckpt


python test_OPDMulti_better.py test \
    --config config/opd_train.yaml \
    --config config/opdmulti_train_no_pretrain.yaml \
    --config config/opdmulti_test.yaml \
    --ckpt_path /cluster/project/cvg/students/andrye/experiments/OPDMulti_v14_no_pretrain/best-epoch=22-val/loss_total=0.9316.ckpt


python test_SF3D_better.py test \
    --config config/sf3d_train.yaml \
    --config config/sf3d_test.yaml \
    --ckpt_path /cluster/project/cvg/students/andrye/experiments/SF3D_V13/best-epoch=17-val/loss_total=0.7952.ckpt

python train_SF3D_better.py 

cp -r /cluster/scratch/andrye/SF3D_lmdb_clean/data.lmdb /dev/shm/data.lmdb


python train_SF3D_better.py fit --config config/sf3d_train.yaml

python test_SF3D_better.py test \
    --config config/sf3d_train.yaml \
    --config config/sf3d_test.yaml \
    --ckpt_path /cluster/scratch/andrye/experiments/SF3D_V15/best-epoch=17-val/loss_total=0.7886.ckpt


python run_single_inference.py \
    --ckpt_path /cluster/scratch/andrye/experiments/SF3D_V15/best-epoch=17-val/loss_total=0.7886.ckpt\
    --data_dir /cluster/scratch/andrye/SF3D_lmdb_clean \
    --text_description "open the cabinet door" \
    --output_dir "my_inference_results" \
    --image_path "/cluster/scratch/andrye/SF3D_lmdb_clean/images/423966_42898337_506769.896.jpg"

python run_single_inference.py \
    --ckpt_path /cluster/scratch/andrye/experiments/SF3D_V15/best-epoch=17-val/loss_total=0.7886.ckpt\
    --data_dir /cluster/scratch/andrye/SF3D_lmdb_clean \
    --text_description "open the window" \
    --output_dir "inf_window" \
    --image_path "/cluster/scratch/andrye/SF3D_lmdb_clean/images/423966_42898337_506769.896.jpg"


python run_single_inference.py \
    --ckpt_path /cluster/scratch/andrye/experiments/SF3D_V15/best-epoch=17-val/loss_total=0.7886.ckpt\
    --data_dir /cluster/scratch/andrye/SF3D_lmdb_clean \
    --text_description "close the curtain" \
    --output_dir "inf_curtain" \
    --image_path "/cluster/scratch/andrye/SF3D_lmdb_clean/images/423966_42898337_506769.896.jpg"


python run_single_inference.py \
    --ckpt_path /cluster/scratch/andrye/experiments/SF3D_V15/best-epoch=17-val/loss_total=0.7886.ckpt\
    --data_dir /cluster/scratch/andrye/SF3D_lmdb_clean \
    --text_description "open the right cabinet door" \
    --output_dir "inf_right_cabinet" \
    --image_path "/cluster/scratch/andrye/SF3D_lmdb_clean/images/423966_42898337_506769.896.jpg"


python run_single_inference.py \
    --ckpt_path /cluster/scratch/andrye/experiments/SF3D_V15/best-epoch=17-val/loss_total=0.7886.ckpt\
    --data_dir /cluster/scratch/andrye/SF3D_lmdb_clean \
    --text_description "open the left cabinet door" \
    --output_dir "inf_left_cabinet" \
    --image_path "/cluster/scratch/andrye/SF3D_lmdb_clean/images/423966_42898337_506769.896.jpg"


python run_custom_inference.py \
    --ckpt_path /cluster/scratch/andrye/experiments/SF3D_V15/best-epoch=17-val/loss_total=0.7886.ckpt \
    --image_path /cluster/home/andrye/ml-depth-pro/data/IMG_9414.jpeg \
    --depth_path /cluster/home/andrye/ml-depth-pro/output/IMG_9414.npz \
    --text_description "open the door" \
    --output_dir "my_custom_results"

python test_OPDReal_better.py test \
    --config config/opd_train.yaml \
    --config config/opdreal_train.yaml \
    --config config/opdreal_test.yaml \
    --ckpt_path /cluster/project/cvg/students/andrye/experiments/OPDReal_v17/best-epoch=25-val/loss_total=0.5977.ckpt




/cluster/scratch/andrye/SF3D_lmdb_clean/data.lmdb


python test_OPDMulti_better.py test \
    --config config/opd_train.yaml \
    --config config/opdmulti_train_no_pretrain.yaml \
    --config config/opdmulti_test.yaml \
    --ckpt_path /cluster/project/cvg/students/andrye/experiments/OPDMulti_v14_no_pretrain/best-epoch=22-val/loss_total=0.9316.ckpt



python test_OPDMulti_better.py test \
    --config config/opd_train.yaml \
    --config config/opdmulti_train.yaml \
    --config config/opdmulti_test.yaml \
    --ckpt_path /cluster/project/cvg/students/andrye/experiments/OPDMulti_v12/best-epoch=11-val/loss_total=0.7877.ckpt