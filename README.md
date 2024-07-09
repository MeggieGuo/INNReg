# INNReg
Codes for “Unsupervised Multi-modal Medical Image Registration via Invertible Translation”.

-[paper](https://eccv2024.ecva.net/)


## Abstract

In medical imaging, the alignment of multi-modal images plays a critical role in providing comprehensive information for image-guided therapies. Despite its importance, multi-modal image registration poses significant challenges due to the complex and often unknown spatial relationships between different image modalities. To address this, we introduce a novel unsupervised translation-based multi-modal registration method, termed Invertible Neural Network-based Registration (INNReg). INNReg consists of an image-to-image translation network that converts multi-modal images into mono-modal counterparts and a registration network that uses the translated mono-modal images to align the multi-modal images. Specifically, to ensure the preservation of geometric consistency after image translation, we introduce an Invertible Neural Network (INN) that leverages a dynamic depthwise convolution-based local attention mechanism. Additionally, we design a novel barrier loss function based on Normalized Mutual Information to impose constraints on the registration network, which enhances the registration accuracy. The superior performance of INNReg is demonstrated through experiments on two public multi-modal medical image datasets, including MRI T1/T2 and MRI/CT pairs.

## Training
Download the dataset:

MRI T1-T2, created from [BraTS 2023](https://www.synapse.org/Synapse:syn53708126/wiki/626320)

MR-CT, created from [Harward](http://www.med.harvard.edu/AANLIB/home.html)

Run `train.py`

## Test
To test the INNReg, run `test.py`

## Citation

```
@InProceedings{
}
```

