# ACC-UNet implementation

Our current implementation is ACC-UNet is computationally expensive due to concat operaions. This has been a pilot study of our ideas and we are optimizing our model further.

We are sharing several variants of the model. Please use the model that applies best for your application.

The model ACC_Unet_Lite is the most lightweight model, which ignores the MLFC block. We have found that this model performs quite well comparatively, particularly for small datasets.