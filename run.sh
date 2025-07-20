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
