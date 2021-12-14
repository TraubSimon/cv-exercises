# Num updates experiment
only run it for default glass_blur severity 1
python run_resnet.py --apply_bn --num_bn_updates x --batch_size 16
## --num_bn_updates 10
Validation complete. Loss 1.772994 accuracy 59.04%
## --num_bn_updates 50
Validation complete. Loss 2.039543 accuracy 56.03%
## --num_bn_updates 100
Validation complete. Loss 1.926813 accuracy 57.56%

# Takehome:
We perform the update on samples of the validation set. The more updates we perform, the more samples we see. Surprisingly, few updates already give a good performance and more do not seem to help. This should be checked for other corruption types