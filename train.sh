cp -r $WORK/sf3d_processed/data.lmdb /dev/shm/data.lmdb

python train_SF3D_better.py fit --config config/sf3d_train.yaml