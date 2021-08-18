# Multi-Modal Self-Supervision using GDT and StiCa

This is an official pytorch implementation of papers: 
[Multi-modal Self-Supervision from Generalized Data Transformations
](https://arxiv.org/abs/2003.04298) and [Space-Time Crop & Attend: Improving Cross-modal Video Representation Learning](https://arxiv.org/abs/2103.10211). 
In this repository, we provide PyTorch code for pretraining and testing our proposed GDT and StiCa models.

If you find GDT and STiCA useful in your research, please use the following BibTeX entries for citation.

```BibTeX

@misc{patrick2020multimodal,
      title={Multi-modal Self-Supervision from Generalized Data Transformations}, 
      author={Mandela Patrick and Yuki M. Asano and Polina Kuznetsova and Ruth Fong and João F. Henriques and Geoffrey Zweig and Andrea Vedaldi},
      year={2021},
      booktitle={International Conference on Computer Vision (ICCV)},
}

@misc{m2021spacetime,
    title={Space-Time Crop & Attend: Improving Cross-modal Video Representation Learning},
    author={Mandela Patrick and Yuki M. Asano and Bernie Huang and Ishan Misra and Florian Metze and Joao Henriques and Andrea Vedaldi},
    year={2021},
    booktitle={International Conference on Computer Vision (ICCV)},
}
```
## Highlights

**(1) GDT: Formulate and generalize most pretext tasks in a NCE objective.** 

Using this formulation, we test various pretext tasks previously unexplored and achieve SOTA downstream performance. 

**(2) STiCA: Importance of incorporating within-modal invariance in cross-modal learning**

We show how to efficiently incorporate within-modal invariance learning using feature crops and achieve SOTA downstream performance.

## Model Zoo

We provide GDT models pretrained on Kinetics-400 (K400), HowTo100M (HT100M), and Instagram-65M (IG65M) datasets, and StiCa models pretrained on Kinetics-400 (K400).

| name | dataset | # of frames | spatial crop | HMDB51 Top1 | UCF101 Top1 | url |
| --- | --- | --- | --- | --- | --- | --- |
| GDT | K400 | 30 | 112 | 62.3 | 90.9 | [model](https://dl.fbaipublicfiles.com/GDT/gdt_K400.pth) |
| GDT | HT100M | 30 | 112 | 94.1 | 67.4 | [model](https://dl.fbaipublicfiles.com/GDT/gdt_HT100M.pth) |
| GDT | IG65M | 30 | 112 | 72.8 | 95.2 | [model](https://dl.fbaipublicfiles.com/GDT/gdt_IG65M.pth) |

| name | dataset | # of frames | spatial crop | UCF101 Top1 | HMDB51 Top1 | url |
| --- | --- | --- | --- | --- | --- | --- |
| STiCA | K400 | 60 | 112 | 67.0 | 93.1 | [Coming Soon](XX) |

## Installation

This repo was tested with Ubuntu 16.04.5 LTS, Python 3.7.5, PyTorch 1.3.1, Torchvision 0.4.1, and CUDA 10.0. 

### Step 1

- Clone this repo to your local machine

### Step 2

- Install required packages using `conda env create -f environment.yml`

### Step 3

- Activate conda environment using `conda activate GDT`

### Step 4

- Install kornia library `pip install kornia==0.1.4` 

### Step 5

- See below for how to pretrain GDT / StiCa or benchmark pretrained models

## Data Preperation

For Kinetics-400/600, HMDB-51 and UCF-101 datasets:
<ol>
<li>Ensure all datasets are in the format: </li>

```
$ROOT_DIR/$SPLIT/$CLASS/*
```

</ol>

To prepare How-To-100M dataset, do the following:
<ol>
<li>Download the word2vec matrix and dictionary, unzip the file, and place in <em>datasets/data</em> folder.</li>

```
wget https://www.rocq.inria.fr/cluster-willow/amiech/word2vec.zip
unzip word2vec.zip
mv word2vec.pth datasets/data/word2vec.pth 
```

<li>Download the csv files of captions.</li>

```
wget https://www.rocq.inria.fr/cluster-willow/amiech/howto100m/howto100m_captions.zip
unzip howto100m_captions.zip
```

<li>Download the preprocessed HowTo100M videos (12TB in total) by filling this Google form: https://forms.gle/hztrfnFQUJWBtiki8.</li>
</ol>


## Usage

### GDT pretraining
To pretrain audio-visual GDT on K-400

**Multi-node distributed training with SLURM cluster:**
```
sbatch pretraining_scripts/pretrain_gdt_k400.sh ${HYPOTHESIS_DESC} ${HYPOTHESIS} 
```

**Single-node distributed training:**
```
python -m torch.distributed.launch --master_port=$RANDOM --nproc_per_node=2 --use_env main_gdt.py --batch_size $BS --lr $LR --hypothesis {1,2,3,4,5,6,7,8,9}
```

To pretrain video-text GDT on HT100M

**Multi-node training with SLURM cluster:**
```
sbatch pretraining_scripts/pretrain_gdt_ht100m.sh ${HYPOTHESIS_DESC} ${HYPOTHESIS} 
```

**Single-node distributed training:**
```
python -m torch.distributed.launch --master_port=$RANDOM --nproc_per_node=2 --use_env main_gdt.py --batch_size $BS --lr $LR --hypothesis {1,2,3,4,5,6,7,8,9} --dataset ht100m --decode_audio False --model vid_text_gdt --sample_rate 2
```

$HYPOTHESIS refers to the hypotheses explored in GDT. We experiment with the following:
 ```
1 - cross-modal baseline (cross_modal_baseline)
2 - variant to time reversal (v_reversal)
3 - invariant to time reversal (i_reversal)
4 - variant to time shift (v_shift)
5 - invariant to time shift (i_shift)
6 - variant to time reversal and variant to time shift (v_reversal_v_shift)
7 - invariant to time reversal, variant to time shift (i_reversal_v_shift)
8 - variant to time reversal, and invariant to time shift (v_reversal_i_shift)
9 - invariant to time reversal, invariant to time shift (i_reversal_i_shift)
```

Please modify the following in SLURM script:
- SBATCH directives (e.g. partition, nodes, constraint,)
- SAV_FOLDER
- --root_dir (path of K-400 / HT100M train directory)


All experiments were run with 8 nodes (64 GPUs, volta32). Please scale batch-size and learning-rate appropriately.

### STiCA pretraining
To pretrain audio-visual STiCA on K-400

**Multi-node training with SLURM cluster:**
```
sbatch scripts/pretrain_stica.sh $NUM_FRAMES $AUD_NUM_SEC $NUM_LARGE_CROPS $NUM_SMALL_CROPS $NUM_SMALL_TCROPS $NUM_LARGE_TCROPS $NUM_LAYER
```

**Single-node distributed training:**
```
python -m torch.distributed.launch --master_port=$RANDOM --nproc_per_node=2 --use_env main_stica.py --batch_size $BS --base_lr $LR
```

**Hyper-parameters:**
 ```
NUM_FRAMES - number of frames (e.g. 30)
AUD_NUM_SEC - number of seconds (30f: 1sec, 60f: 2s)
NUM_LARGE_CROPS - num of large feature spatial crops (e.g. 2)
NUM_SMALL_CROPS - num of small feature spatial crops (e.g. 4)
NUM_SMALL_TCROPS - num of large feature spatial crops (e.g. 1)
NUM_LARGE_TCROPS - num of small feature spatial crops (e.g. 2)
NUM_LAYER - num of transformer pooling layers (0 == GAP, >1 is num. of transformer layers)
e.g. sbatch scripts/pretrain_stica.sh 30 1 2 4 1 2 0
```

Please modify the following in SLURM script:
- SBATCH directives (e.g. partition, nodes, constraint,)
- SAV_FOLDER
- --root_dir (path of K-400 / HT100M train directory)


All experiments were run with 8 nodes (64 GPUs, volta32). Please scale batch-size and learning-rate appropriately.

## Benchmarking

To evaluate pretraining on video action recognition on UCF-101 and HMDB-51 datasets,

Locally:
```
python3 eval_video.py --dataset {ucf101, hmdb51} --fold {1,2,3} --weights-path {WEIGHTS_PATH} --model ${vid_text_gdt, stica, av_gdt}
```

On SLURM:
```
bash scripts/eval.sh ${WEIGHTS_PATH} ${OUTPUT_DIR} ${CKPT_NUM} ${CLIP_LEN} ${vid_text_gdt, stica, av_gdt} ${1, 2, 3}
``` 

Modify --root_dir, --ucf101-annotation-path, and --hmdb51-annotation-path in eval_video.py.

## License

The majority of this work is licensed under [CC-NC 4.0 International license](LICENSE).

## Contributing

We actively welcome your pull requests. Please see [CONTRIBUTING.md](CONTRIBUTING.md) and [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) for more info.
