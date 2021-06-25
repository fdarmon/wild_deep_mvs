# Deep MVS gone wild

Pytorch implementation of "Deep MVS gone wild" ([Paper](https://arxiv.org/pdf/2104.15119) | [website](https://imagine.enpc.fr/~darmonf/wild_deep_mvs))

This repository provides the code to reproduce the experiments of the paper. It implements extensive comparison of Deep MVS architecture, training data and supervision.

If you find this repository useful for your research, please consider citing
```
@article{
  author    = {Darmon, Fran{\c{c}}ois  and
               Bascle, B{\'{e}}n{\'{e}}dicte  and
               Devaux, Jean{-}Cl{\'{e}}ment  and
               Monasse, Pascal  and
               Aubry, Mathieu},
  title     = {Deep Multi-View Stereo gone wild},
  year      = {2021},
  url       = {https://arxiv.org/abs/2104.15119},
}
```

## Installation

* Python packages: see `requirements.txt`

* Fusibile:
```shell
git clone https://github.com/YoYo000/fusibile 
cd fusibile
cmake .
make .
ln -s EXE ./fusibile
 ```

* COLMAP: see the [github repository](https://github.com/colmap/colmap) for installation details then link colmap executable with `ln -s COLMAP_DIR/build/src/exe/colmap colmap`

## Training

You may find all the pretrained models [here](https://drive.google.com/file/d/1HaQQQ9pkf5DuYQXffIrxoVCTpwLF5YdK/view?usp=sharing) (120 Mo) or alternatively you can train models using the following instructions.  

### Data
Download the following data and extract to folder `datasets` 

  * [DTU training](https://drive.google.com/file/d/1eDjh-_bxKKnEuz5h-HXS7EDJn59clx6V/view) (19 Go)
  * [BlendedMVS](https://1drv.ms/u/s!Ag8Dbz2Aqc81gVDgxb8MDGgoV74S?e=hJKlvV) (27.5 Go)
  * Megadepth: [MegadepthV1](https://www.cs.cornell.edu/projects/megadepth/dataset/Megadepth_v1/MegaDepth_v1.tar.gz) (199 Go)
[Geometry](https://drive.google.com/file/d/1UYXIS7U8iWzctwCzSQ5CcawK_M5NgZwu/view?usp=sharing) (8 Go)

The directory structure should be as follow:
```
datasets
├─ blended
├─ dtu_train
├─ MegaDepth_v1
├─ undistorted_md_geometry
```

The data is already preprocessed for DTU and BlendedMVS. For MegaDepth, run `python preprocess.py` for generating the training data.

### Script

The training script is `train.py`, launch `python train.py --help` for all the options. For example 

* `python train.py --architecture vis_mvsnet --dataset md --supervised --logdir best_sup --world_size 4 --batch_size 4` 
  for training the best performing setup for images in the wild.
* `python train.py --architecture mvsnet-s --dataset md --unsupervised --upsample --occ_masking --epochs 5 --lrepochs 4:10 --logdir best_unsup --world_size 3` for the best unsupervised model. 

The models are saved in folder `trained_models`

## Evaluations
We provide code for both depthmap evaluation and 3D reconstruction evaluation

### Data
Download the following links and extract them to `datasets`
  * [BlendedMVS](https://1drv.ms/u/s!Ag8Dbz2Aqc81gVDgxb8MDGgoV74S?e=hJKlvV) (27.5 GB) same link as BlendedMVS training data
  * [YFCC depth maps](https://drive.google.com/file/d/12mD0X3YIqsiMsX7jkUQ5-D38D_rojg4Q/view?usp=sharing) (1.1Go)
  * DTU MVS benchmark: Create directory `datasets/dtu_eval` and extract the following files
    * [Images](https://drive.google.com/file/d/135oKPefcPTsdtLRzoDAQtPpHuoIrpRI_/view) (500Mo), rename it as `images` folder
    * [Ground truth](http://roboimagedata2.compute.dtu.dk/data/MVS/Points.zip) (6.3Go)
    * [evaluation files](http://roboimagedata2.compute.dtu.dk/data/MVS/SampleSet.zip) (6.3Go), the evaluation only need `ObsMask` folder

    In the end the folder structure should be
    ```
    datasets
    ├─ dtu_eval
        ├─ ObsMask
        ├─ images
        ├─ Points
            ├─ stl
    ```
  * [YFCC 3D reconstruction](https://drive.google.com/file/d/1PCEYTPU4V7kXyGc-qfH5eNxQJIIcsWtq/view?usp=sharing) (1.5Go)

  
### Depthmap evaluation
`python depthmap_eval.py --model MODEL --dataset DATA` 
* `MODEL` is the name of a folder found in `trained_models`
* `DATA` is the evaluation dataset, either `yfcc` or `blended`

### 3D reconstruction

See `python reconstruction_pipeline.py --help` for a complete list of parameters for 3D reconstruction.
For running the whole evaluation for a trained model with the parameters used in the paper, run
* `scripts/eval3d_dtu.sh --model MODEL (--compute_metrics)` for DTU evaluation
* `scripts/eval3d_yfcc.sh --model MODEL (--compute_metrics)` for YFCC 3D evaluation

The reconstruction will be located in `datasets/dtu_eval/Points` or `datasets/yfcc_data/Points`

## Acknowledgments

This repository is inspired by [MVSNet_pytorch](https://github.com/xy-guo/MVSNet_pytorch) and [MVSNet](https://github.com/YoYo000/MVSNet)
repositories. We also adapt the official implementations of [Vis_MVSNet](https://github.com/jzhangbs/Vis-MVSNet) and [CVP_MVSNet](https://github.com/JiayuYANG/CVP-MVSNet).

## Copyright

```
Deep MVS Gone Wild All rights reseved to Thales LAS and ENPC.

This code is freely available for academic use only and Provided “as is” without any warranty.

Modification are allowed for academic research provided that the following conditions are met :
  * Redistributions of source code or any format must retain the above copyright notice and this list of conditions.
  * Neither the name of Thales LAS and ENPC nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
```
 
