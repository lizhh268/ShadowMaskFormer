# ShadowMaskFormer
A transformer-based approach for image shadow removal.
This repository includes code for the following paper:

ShadowMaskFormer: Mask Augmented Patch Embedding for Shadow Removal

# Training Environment
We test the code on PyTorch 1.10.2 + CUDA 11.3 + cuDNN 8.2.0.

1. Create a new conda environment
```
conda create -n shadowmaskformer python=3.7
conda activate shadowmaskformer
```

2. Install dependencies
```
conda install pytorch=1.10.2 torchvision torchaudio cudatoolkit=11.3 -c pytorch
pip install -r requirements.txt
```

# Insturcions
Please execute the following instructions to configure the parameters for running the program:

1. Model Training
python train.py --model (model name) --dataset (dataset name) --exp (exp name)
e.g.: python train.py --model shadowmaskformer-b --dataset ISTD --exp istd

2. Model Testing
python test.py --model (model name) --dataset (dataset name) --exp (exp name)
e.g.: python test.py --model shadowmaskformer-b --dataset ISTD --exp istd
