# Reproducibility

In our code, we carefully set the random seed and set cudnn as 'deterministic' mode to eliminate the randomness. 
However, there still exsist some factors which may cause different training results, e.g., the cuda version, GPU types, the number of GPUs and etc. The GPU used in our experiments mostly is NVIDIA A6000 (48G) and the cuda version is 11.3.

Especially for multi-GPU cases, the upsampling operation has big problems with randomness.
See https://pytorch.org/docs/stable/notes/randomness.html for more details.

Here, we are providing the dataset splits we used in our experiments and also the training logs.