# FlowNetC/FlowNetS Training and Evaluation

This repository contains code for training and evaluating FlowNetC/FlowNetS models.

## Setup

The code was tested with python 3.8 and PyTorch 1.9. To install the requirements, run:
```bash
pip install -r requirements.txt
```

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

Pre-training on the FlyingThings dataset:
```bash
python train.py --output /your/output/directory --model FlowNetC --cuda_corr
```

The cuda_corr flag is optional but recommended. It significantly speeds up training time but requires compilation of
the CUDA correlation layer as described above. 
Checkpoints and tensorboard logs will be written to the specified output directory.
Self-supervised training is possible with the flag --photometric (and optionally the flag --smoothness_loss).

Pre-training on the FlyingChairs dataset:
```bash
python train.py --output /your/output/directory --model FlowNetC --cuda_corr --dataset FlyingChairs
```

Pre-training on FlyingChairs+FlyingThings3D:
```bash
python train.py --output /your/output/directory --model FlowNetC --cuda_corr --dataset FlyingChairs --iterations 300000
python train.py --output /your/output/directory --model FlowNetC --cuda_corr --dataset FlyingThings3D --restore /path/to/chkpt/checkpoint-train-iter-000300000.pt --completed_iterations 300000 --iterations 600000
```

##### FlowNetS
Pre-training on the FlyingThings dataset:
```bash
python train.py --output /your/output/directory --model FlowNetS
```

Pre-training on FlyingChairs and on FlyingChairs+FlyingThings3D works as described above.

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

### Results
Finetuned models are provided under /project/cv-ws2122/shared-data1/OpticalFlowFinetunedModels/

- training FlowNetS on Chairs for 600k iterations and evaluating on SintelFull: AEPE=4.94
- training FlowNetC on Chairs for 600k iterations and evaluating on SintelFull: AEPE=4.16
- training FlowNetS on Chairs for 600k iterations and evaluating on Sintel (our test split): AEPE=0.60
- training FlowNetS on Chairs for 600k iterations + 100k iterations on Sintel (our train split) with photometric+smoothness loss and evaluating on Sintel (our test split): AEPE=0.57
- training FlowNetS on Chairs for 600k iterations + 100k iterations on Sintel (our train split) supervised and evaluating on Sintel (our test split): AEPE=0.54
