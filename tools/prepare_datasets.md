## Prepare datasets

In our paper, we conduct experiments on three common-used datasets, including Ref-COCO, Ref-COCO+ and G-Ref.

### 1. COCO 2014

The data could be found at [here](https://cocodataset.org/#download). Please run the following commands to download.

```shell
# download
mkdir datasets && cd datasets
wget http://images.cocodataset.org/zips/train2014.zip

# unzip
unzip train2014.zip -d images/ && rm train2014.zip

```

```shell
export PROJECT_ROOT=/cluster/home/andrye/CRIS.pytorch
export DATA_ROOT=/cluster/project/cvg/data/refcoco
```

### 2. Ref-COCO

The data could be found at [here](https://github.com/lichengunc/refer). Please run the following commands to download and convert.

```shell
# download
wget https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco.zip

# unzip
unzip refcoco.zip && rm refcoco.zip

# convert
python $PROJECT_ROOT/tools/data_process.py --data_root $DATA_ROOT --output_dir $DATA_ROOT --dataset refcoco --split unc --generate_mask

# lmdb
python $PROJECT_ROOT/tools/folder2lmdb.py -j $DATA_ROOT/anns/refcoco/val.json -i $DATA_ROOT/images/train2014/ -m $DATA_ROOT/masks/refcoco -o $DATA_ROOT/lmdb/refcoco
python $PROJECT_ROOT/tools/folder2lmdb.py -j $DATA_ROOT/anns/refcoco/testA.json -i $DATA_ROOT/images/train2014/ -m $DATA_ROOT/masks/refcoco -o $DATA_ROOT/lmdb/refcoco
python $PROJECT_ROOT/tools/folder2lmdb.py -j $DATA_ROOT/anns/refcoco/train.json -i $DATA_ROOT/images/train2014/ -m $DATA_ROOT/masks/refcoco -o $DATA_ROOT/lmdb/refcoco
python $PROJECT_ROOT/tools/folder2lmdb.py -j $DATA_ROOT/anns/refcoco/testB.json -i $DATA_ROOT/images/train2014/ -m $DATA_ROOT/masks/refcoco -o $DATA_ROOT/lmdb/refcoco

# clean
rm -r refcoco

```

### 3. Ref-COCO+

The data could be found at [here](https://github.com/lichengunc/refer). Please run the following commands to download and convert.

```shell
# download
wget https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco+.zip

# unzip
unzip refcoco+.zip && rm refcoco+.zip

# convert
python $PROJECT_ROOT/tools/data_process.py --data_root $DATA_ROOT --output_dir $DATA_ROOT --dataset refcoco+ --split unc --generate_mask

# lmdb
python $PROJECT_ROOT/tools/folder2lmdb.py -j $DATA_ROOT/anns/refcoco+/train.json -i $DATA_ROOT/images/train2014/ -m $DATA_ROOT/masks/refcoco+ -o $DATA_ROOT/lmdb/refcoco+
python $PROJECT_ROOT/tools/folder2lmdb.py -j $DATA_ROOT/anns/refcoco+/val.json -i $DATA_ROOT/images/train2014/ -m $DATA_ROOT/masks/refcoco+ -o $DATA_ROOT/lmdb/refcoco+
python $PROJECT_ROOT/tools/folder2lmdb.py -j $DATA_ROOT/anns/refcoco+/testA.json -i $DATA_ROOT/images/train2014/ -m $DATA_ROOT/masks/refcoco+ -o $DATA_ROOT/lmdb/refcoco+
python $PROJECT_ROOT/tools/folder2lmdb.py -j $DATA_ROOT/anns/refcoco+/testB.json -i $DATA_ROOT/images/train2014/ -m $DATA_ROOT/masks/refcoco+ -o $DATA_ROOT/lmdb/refcoco+

# clean
rm -r refcoco+

```

### 4. Ref-COCOg

The data could be found at [here](https://github.com/lichengunc/refer). Please run the following commands to download and convert.
(Note that we adopt two different splits of this dataset, 'umd' and 'google'.)

```shell
# download
wget https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcocog.zip

# unzip
unzip refcocog.zip && rm refcocog.zip

# convert
python $PROJECT_ROOT/tools/data_process.py --data_root $DATA_ROOT --output_dir $DATA_ROOT --dataset refcocog --split umd --generate_mask  # umd split
mv $DATA_ROOT/anns/refcocog $DATA_ROOT/anns/refcocog_u
mv $DATA_ROOT/masks/refcocog $DATA_ROOT/masks/refcocog_u

python $PROJECT_ROOT/tools/data_process.py --data_root $DATA_ROOT --output_dir $DATA_ROOT --dataset refcocog --split google --generate_mask  # google split
mv $DATA_ROOT/anns/refcocog $DATA_ROOT/anns/refcocog_g
mv $DATA_ROOT/masks/refcocog $DATA_ROOT/masks/refcocog_g

# lmdb
python $PROJECT_ROOT/tools/folder2lmdb.py -j $DATA_ROOT/anns/refcocog_u/train.json -i $DATA_ROOT/images/train2014/ -m $DATA_ROOT/masks/refcocog_u -o $DATA_ROOT/lmdb/refcocog_u
python $PROJECT_ROOT/tools/folder2lmdb.py -j $DATA_ROOT/anns/refcocog_u/val.json -i $DATA_ROOT/images/train2014/ -m $DATA_ROOT/masks/refcocog_u -o $DATA_ROOT/lmdb/refcocog_u
python $PROJECT_ROOT/tools/folder2lmdb.py -j $DATA_ROOT/anns/refcocog_u/test.json -i $DATA_ROOT/images/train2014/ -m $DATA_ROOT/masks/refcocog_u -o $DATA_ROOT/lmdb/refcocog_u

python $PROJECT_ROOT/tools/folder2lmdb.py -j $DATA_ROOT/anns/refcocog_g/train.json -i $DATA_ROOT/images/train2014/ -m $DATA_ROOT/masks/refcocog_g -o $DATA_ROOT/lmdb/refcocog_g
python $PROJECT_ROOT/tools/folder2lmdb.py -j $DATA_ROOT/anns/refcocog_g/val.json -i $DATA_ROOT/images/train2014/ -m $DATA_ROOT/masks/refcocog_g -o $DATA_ROOT/lmdb/refcocog_g

rm -r refcocog

```

### 5. Datasets struture

After the above-mentioned commands, the strutre of the dataset folder should be like:

```none
datasets
├── anns
│   ├── refcoco
│   │   ├── xxx.json
│   ├── refcoco+
│   │   ├── xxx.json
│   ├── refcocog_g
│   │   ├── xxx.json
│   ├── refcocog_u
│   │   ├── xxx.json
├── images
│   ├── train2014
│   │   ├── xxx.jpg
├── lmdb
│   ├── refcoco
│   │   ├── xxx.lmdb
│   │   ├── xxx.lmdb-lock
│   ├── refcoco+
│   │   ├── xxx.lmdb
│   │   ├── xxx.lmdb-lock
│   ├── refcocog_g
│   │   ├── xxx.lmdb
│   │   ├── xxx.lmdb-lock
│   ├── refcocog_u
│   │   ├── xxx.lmdb
│   │   ├── xxx.lmdb-lock
├── masks
│   ├── refcoco
│   │   ├── xxx.png
│   ├── refcoco+
│   │   ├── xxx.png
│   ├── refcocog_g
│   │   ├── xxx.png
│   ├── refcocog_u
│   │   ├── xxx.png

```