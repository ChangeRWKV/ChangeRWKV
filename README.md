<h1 align="center">ChangeRWKV</h1>

<h3 align="center">Beyond Quadratic: Linear-Time Change Detection with RWKV</h3>

## 🔭Overview

![ChangeRWKV](changerwkv.png)

## 🗝️Quick start on LEVIR dataset

1. Data preparation:

   - Download LEVIR dataset from [Here](https://justchenhao.github.io/LEVIR/) (Official) or [AI Studio](https://aistudio.baidu.com/datasetdetail/53795) (Spare).

   - Crop the images & masks into $256\times 256$ size.

   - Arrange them into the following format:

     ```
     data
     |- LEVIR_ABlabel
        |- A
           |- train_1_0_256_0_256.png
           |- ...
           |- val_1_0_256_0_256.png
           |- ...
           |- test_1_0_256_0_256.png
           |- ...
        |- B
           |- (same as above)
        |- label
           |- (same as above)
        |- list
           |- train.txt
           |- val.txt
           |- test.txt
     ```

     Here, each line of .txt file is a file name, like:

     ```
     train_1_0_256_0_256.png
     train_1_0_256_256_512.png
     train_1_0_256_512_768.png
     train_1_0_256_768_1024.png
     ...
     ```

2. Installation:

   - Clone this repository and navigate to the project directory:

     ```bash
     git clone https://github.com/ChangeRWKV/ChangeRWKV.git
     cd ChangeRWKV
     ```

   - Create and activate a new conda environment:

     ```bash
     conda create -n changerwkv python=3.10
     conda activate changerwkv
     ```

   - Install PyTorch according to CUDA or ROCm version (taking CUDA 12.1 as an example here): 

     ```bash
     pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
     ```

     👉 **Note that** we do not recommend you to use torch versions lower than **2.0.0**.

   - Install other dependencies:

     ```bash
     pip install -r requirements.txt
     ```
   
3. Download Pretrained Weight:

   - Please download the pretrained weight of our [Hierarchical RWKV Encoder](https://github.com/ChangeRWKV/ChangeRWKV/releases/download/v0.1/vrwkv_b.pth) and put it under **project_path/ckpt/**.

4. Train:

   ```bash
   python train.py --model_size 'base' --pretrain_dir 'ckpt/vrwkv_b.pth'
   ```

5. Benchmark:

   - Please download the weight of our [ChangeRWKV](https://github.com/ChangeRWKV/ChangeRWKV/releases/download/v0.1/changerwkv-b_levir.pth) and put it under **project_path/ckpt/**.

   - Test:

     ```bash
     python test.py --model_size 'base' --model_path 'ckpt/changerwkv-b_levir.pth'
     ```

## 🤝Acknowledgments
This project is based on [RWKV-UNet](https://github.com/juntaoJianggavin/RWKV-UNet) and [AFCF3D-Net](https://github.com/wm-Githuber/AFCF3D-Net). Thanks for their excellent works!!
