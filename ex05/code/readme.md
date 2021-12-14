# FlowNetC/FlowNetS Training and Evaluation

This repository contains code for training and evaluating FlowNetC/FlowNetS models.

## Setup

The code was tested with python 3.8 and PyTorch 1.9. To install the requirements, run:
```bash
pip install -r requirements.txt
```

#### Note: 

Since the gcc version on the pool is not compatible yet, please ignore the cuda correlation until next week.


The FlowNetC model can be used with a CUDA correlation layer or a python correlation layer. The CUDA correlation
layer is faster but needs to be precompiled. To compile the CUDA correlation layer, run:
```bash
cd lib/cuda_correlation_package
python setup.py install
```

## Usage

### Training

#### Pre-training

##### FlowNetC

Pre-training is currently supported on the FlyingThings dataset and models can be found at ```/project/cv-ws2122/shared-data1/OpticalFlowPretrainedModels```. Pre-training on FlyingChairs will be added later.
```bash
python train.py --output /your/output/directory --model FlowNetC --cuda_corr
```

The cuda_corr flag is optional but recommended. It significantly speeds up training time but requires compilation of
the CUDA correlation layer as described above. 
Checkpoints and tensorboard logs will be written to the specified output directory.
Self-supervised training is possible with the flag --photometric (and optionally the flag --smoothness_loss).

##### FlowNetS
```bash
python train.py --output /your/output/directory --model FlowNetS
```

#### Fine-tuning

Fine-tuning is currently supported on the Sintel dataset.
##### FlowNetC
```bash
python train.py --output /your/output/directory --model FlowNetC --cuda_corr --dataset Sintel --restore /path/to/chkpt/checkpoint-model-iter-000600000.pt --completed_iterations 600000 --iterations 700000
```
This would fine-tune the model for 100k iterations in supervised mode.

For self-supervised fine-tuning with a photometric loss, run:
```bash
python train.py --output /your/output/directory --model FlowNetC --cuda_corr --dataset Sintel --restore /path/to/chkpt/checkpoint-model-iter-000600000.pt --completed_iterations 600000 --iterations 700000 --photometric
```

To include the smoothness loss, run:
```bash
python train.py --output /your/output/directory --model FlowNetC --cuda_corr --dataset Sintel --restore /path/to/chkpt/checkpoint-model-iter-000600000.pt --completed_iterations 600000 --iterations 700000 --photometric --smoothness_loss
```

##### FlowNetS
```bash
python train.py --output /your/output/directory --model FlowNetS --dataset Sintel --restore /path/to/chkpt/checkpoint-model-iter-000600000.pt --completed_iterations 600000 --iterations 700000
```

For self-supervised fine-tuning with a photometric loss, run:
```bash
python train.py --output /your/output/directory --model FlowNetS --dataset Sintel --restore /path/to/chkpt/checkpoint-model-iter-000600000.pt --completed_iterations 600000 --iterations 700000 --photometric
```

To include the smoothness loss, run:
```bash
python train.py --output /your/output/directory --model FlowNetS --dataset Sintel --restore /path/to/chkpt/checkpoint-model-iter-000600000.pt --completed_iterations 600000 --iterations 700000 --photometric --smoothness_loss
```

### Evaluation

#### Evaluate FlowNetC on FlyingThings
```bash
python eval.py --output /your/output/directory --model FlowNetC --cuda_corr --dataset FlyingThings3D --restore /path/to/chkpt/checkpoint-model-iter-000600000.pt
```
Evaluation results will be written to the specified output directory. Qualitative results are written to Tensorboard.
Again, the cuda_corr flag is optional.

#### Evaluate FlowNetS on FlyingThings
```bash
python eval.py --output /your/output/directory --model FlowNetS --dataset FlyingThings3D --restore /path/to/chkpt/checkpoint-model-iter-000600000.pt
```

#### Evaluate FlowNetC on Sintel
```bash
python eval.py --output /your/output/directory --model FlowNetC --cuda_corr --dataset Sintel --restore /path/to/chkpt/checkpoint-model-iter-000600000.pt
```

#### Evaluate FlowNetS on Sintel
```bash
python eval.py --output /your/output/directory --model FlowNetS --dataset Sintel --restore /path/to/chkpt/checkpoint-model-iter-000600000.pt
```
