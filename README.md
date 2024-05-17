# nnU-Net.wSD

This is a direct extension of the work by [soumbane's DualSelfDistillation](https://github.com/soumbane/DualSelfDistillation) applied to medical image segmentation of the cardiac substructures from low-field ViewRay MR-Linac volumes as presented in the International Journal of Radiation Oncology $\bullet$ Biology $\bullet$ Physics paper: [_Enhancing Precision in Cardiac Segmentation for MR-Guided Radiation Therapy through Deep Learning_]().


We extend the publicly available state-of-the-art deep learning (DL) framework to incorporate dual self-distillation along the encoding and decoding branches.


---
## Contents:
This github repository consists of two main parts: 
1) The direct, hard coded model used in the cardiac segmentation paper
2) A more user-friendly, flexible version of the DynU-Net backbone



## Getting started with nnU-Net.wSD:
### Model Output during Training:
In short, at every level of both the encoding and decoding branches, the output is collected and upsampled to match the size of the model's main output. The last encoder and decoder respectively form the teacher while all the other levels are the students. KLDivergence losses are calculated between each student and the teacher to enable self-distillation along each branch.

While `network.training == True`, the output of `network.forward(x)` is a `dict` option to make it user-friendly with the following structure:
```
out = {
  'out': torch.Tensor, # The main output of the model
  'encoder': { # Outputs from the encoding branch
     'teacher': torch.Tensor, # output from the deepest / last encoder
     'students': list[torch.Tensor] # outputs from all the students in the encoding branch
  },
  'decoder': { # Outputs from the decoding branch
      'teacher': torch.Tensor, # output from the deepest / last decoder
      'students': list[torch.Tensor] # outputs from all the students in the decoding branch
  }
```

### Defining the model
The nnU-Net is based off of the Dynamic U-Net backbone with hyper parameters that are tuned through the nnU-Net's framework. To enable the custom training required to enable self-distillation, the nnU-Net's framework is first utilized to generate the model's parameters. In our work, these parameters were then directly taken and used when defining the model's architecture. This is the recommended approach for defining a model.

### Loss functions
The composite loss during training includes the direct comparison between model output and ground truth, the different levels of super-vision down the decoding branch, as well as the levels of self-distillation along both the encoding and decoding branches.

The included class `Model.Losses.SelfDistil_Losses.CompositeLoss` was built to taken in the output dictionary as defined above and calculate the described losses.

