# Batch size experiment
only run it for default glass_blur severity 1
## bs = 1
python run_resnet.py --apply_bn --num_bn_updates 50 --batch_size 1
Validation step 100/10000
Validation complete. Loss 3.114642 accuracy 41.85%
## bs = 4
python run_resnet.py --apply_bn --num_bn_updates 50 --batch_size 4
Validation step 100/2500
Validation complete. Loss 2.039543 accuracy 56.03%
## bs = 16
python run_resnet.py --apply_bn --num_bn_updates 50 --batch_size 16
Validation step 100/625
Validation complete. Loss 1.934974 accuracy 57.61%
## bs = 64
python run_resnet.py --apply_bn --num_bn_updates 50 --batch_size 64
Validation step 100/157
Validation complete. Loss 1.933209 accuracy 57.75%

# Takehome:
Since the batchnorm statistics - mean and variance - are computed over the samples in the batch, the larger the batch size the more stable and meaningful they become.