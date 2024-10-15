
<p align="center">

  <h1 align="center">GS^3: Efficient Relighting with Triple Gaussian Splatting</h1>
  <p align="center">
    <a href="https://github.com/RupertPaoZ/"><strong>Zoubin Bi</strong></a>
    ·
    <a href="https://zyx45889.github.io/"><strong>Yixin Zeng</strong></a>
    ·
    <a href="https://www.chong-zeng.com/"><strong>Chong Zeng</strong></a>
    ·
    <a href=""><strong>Fan Pei</strong></a>
    ·
    <a href="https://f1shel.github.io/"><strong>Xiang Feng</strong></a>
    ·
    <a href="http://kunzhou.net/"><strong>Kun Zhou</strong></a>
    ·
    <a href="https://svbrdf.github.io/"><strong>Hongzhi Wu</strong></a>
  </p>
  <h2 align="center">SIGGRAPH Asia 2024 Conference Papers</h2>
  <div align="center">
    <img src="assets/teaser.jpg">
  </div>

  <p align="center">
  <br>
    <a href="https://gsrelight.github.io/"><strong>Project Page</strong></a>
    |
    <a href="https://gsrelight.github.io/pdfs/GS3.pdf"><strong>Paper</strong></a>
    |
    <a href=""><strong>arXiv</strong></a>
    |
    <a href=""><strong>Data</strong></a>
  </p>
</p>

*We present a spatial and angular Gaussian based representation and a triple splatting process, for real-time, high-quality novel lighting-and-view synthesis from multi-view point-lit input images.*

---

## Setup

### Environment

#### Conda

```bash
conda create --name gs3 python=3.10 pytorch==2.4.1 torchvision==0.19.1 pytorch-cuda=12.4 cuda-toolkit=12.4 cuda-cudart=12.4 -c pytorch -c "nvidia/label/cuda-12.4.0"
conda activate gs3
pip install ninja  # speedup torch cuda extensions compilation
pip install -r requirements.txt
```

#### Docker

We also provide a docker container.

You can use our pre-built docker image.

```bash
docker run -it --gpus all --rm iamncj/gs3:241002
```

Or you can build your own docker image.

```bash
docker build -t gs3:latest .
```

### Usage

We provide a few sample scripts from our paper.

#### Train

For real captured scenes, please use `--cam_opt` and `--pl_opt` to enable camera pose and light optimization.

```bash
bash real_train.sh # real captured scenes
bash syn_train.sh # synthetic scenes
```

#### Test

For real captured scenes, we provide `--valid` and corresponding `.json` file to render a circle view. If you are going to run the test set of real captured scenes, please remember to add `--opt_pose` to use the calibrated poses.

```bash
bash real_render.sh # real captured scenes
bash syn_render.sh # synthetic scenes
```

## Data

We release our data and pretrained models at [huggingface](https://huggingface.co/gsrelight).

> Note: For blender scene rendering, we use the [script](https://github.com/iamNCJ/bpy-helper/tree/main/examples/nrhints-data) from NRHints. For pre-captured scene rendering, we use [Asuna](https://github.com/f1shel/asuna).

You can download the data by running the following command:

```bash
pip install huggingface_hub[hf_transfer]
HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download --repo-type dataset gsrelight/gsrelight-data --local-dir /path/to/data
```

## Citation

Cite as below if you find this repository is helpful to your project:

```
@inproceedings{bi2024rgs,
    title      = {GS\textsuperscript{3}: Efficient Relighting with Triple Gaussian Splatting},
    author     = {Zoubin Bi and Yixin Zeng and Chong Zeng and Fan Pei and Xiang Feng and Kun Zhou and Hongzhi Wu},
    booktitle  = {SIGGRAPH Asia 2024 Conference Papers},
    year       = {2024}
}
```

Some of our dataset are borrowed from [NRHints](https://github.com/iamNCJ/NRHints). Please also cite NRHints if you use those data.

## Acknowledgments

We have intensively borrow codes from [gaussian splatting](https://github.com/graphdeco-inria/gaussian-splatting) and [gsplat](https://github.com/nerfstudio-project/gsplat). We also use [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn) for it's efficient MLP implementation. Many thanks to the authors for sharing their codes.
