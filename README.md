# MR to pCT for TUS

This script produces a pseudo-CT image from a T1-weighted MR image for use in acoustic simulations of transcranial ultrasound stimulation (TUS).


## Platform

Tested on Linux (Ubuntu 20.04.4 LTS) and on macOS Catalina (10.15.7).  
Works with both NVIDIA GPU and CPU-only platforms.


## Dependencies

* [MONAI] (https://monai.io/)
* [ANTsPy] (https://github.com/ANTsX/ANTsPy)


## Instructions

Install dependencies (see above). 

Clone or download python notebook, utility scripts, trained weights, and example dataset. 

In cell #2 of the python notebook mr-to-pct_infer.ipynb, change the path to point to your input MR image, output pCT image, and the location where you saved the trained network weights:
```
# Set data file paths
input_mr_file = '/Users/sitiyaakub/Documents/Analysis/MRtoCT/ForGitHub/sub-test01_t1w.nii'
output_pct_file = '/Users/sitiyaakub/Documents/Analysis/MRtoCT/ForGitHub/sub-test01_pct.nii'
trained_weights = '/Users/sitiyaakub/Documents/Analysis/MRtoCT/ForGitHub/pretrained_net_final_20220825.pth'
```

You may optionally prepare your T1-weighted MR image. If prep_t1 is set to True, the T1-weighted MR image will be bias corrected (using ANTs N4BiasFieldCorrection) and backgound noise outside the head will be masked out.
```
# Do you want to prepare the t1 image? This will perform bias correction and create a head mask
# yes = True, no = False. Output will be saved to <mr_file>_prep.nii
prep_t1 = True
```

You may also optionally produce an example figure of the pCT output. If plot_mrct is set to True, an example figure will be produced. 
```
# Do you want to produce an example plot? yes = True, no = False. 
plot_mrct = True
```

Run notebook.

This will produce the output pCT image in the specified file path.


## Input to network

The software works best for input T1-weighted MR images with the following specifications:
1) scanner: Siemens Prisma 3T
2) acquisition parameters: acquired in sagittal plane, 2100 ms repetition time (TR), 2.26 ms echo time (TE), 900 ms inversion time (TI), 8° flip angle (FA), GRAPPA acceleration factor of 2, and 1 mm<sup>3</sup> voxel size
2) maximum matrix size: 256 x 256 x 256
3) voxel size: 1mm isotropic
4) bias-corrected (e.g. using N4BiasFieldCorrection in ANTs or similar: see https://github.com/ANTsX/ANTs)
5) noise outside the head masked out

The bias correction and noise masking can be optionally applied within the script by setting `prep_t1 = True`.



## Citing this work

The rationale and principle are described in detail in the following paper.

>    Siti N. Yaakub, Tristan White, Eric Kerfoot, Lennart Verhagen, Alexander Hammers, Elsa Fouragnan, 
>    "Pseudo-CTs from T1-weighted MRI for planning of low-intensity transcranial focused ultrasound neuromodulation". (in preparation)

If you use this MR to pCT method in your own work, please acknowledge the script by citing the above, when available.

Please also consider citing ANTsPy and MONAI (see the websites for details).

Feedback welcome at siti.yaakub@plymouth.ac.uk

### TO DO:
1) Upload trained network somewhere.

