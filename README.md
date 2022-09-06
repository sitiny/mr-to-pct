# MR to pCT for TUS

This software produces a pseudo-CT image from a T1-weighted MR image for use in acoustic simulations of transcranial ultrasound stimulation (TUS).


### Publication

The rationale and principle are described in detail in the following paper.

>    Yaakub, S. N., White, T., Kerfoot, E., Verhagen, L., Hammers, A., Fouragnan, E., in prep.
>    Pseudo-CTs from T1-weighted MRI for planning of low-intensity transcranial focused ultrasound neuromodulation.
>    

If you use this software in your own work, please acknowledge the software by citing the above.


### Platform

Tested on Linux (Ubuntu 20.04.4 LTS) and on macOS (version).  
Works with both NVIDIA GPU and CPU-only platforms.


### Dependencies

* [MONAI] (https://monai.io/)


### Instructions

Clone or download python notebook and trained weights. In cell #6, change the path to your input MR and output pCT images, and the trained weights:
```
# set input and output data
input_mr_file = '/home/Researcher/Analysis/mr-to-pct_tests/sub-test01_t1w.nii'
output_pct_file = '/home/Researcher/Analysis/mr-to-pct_tests/sub-test01_pct.nii'

# trained model to load
saved_model = torch.load("/home/Researcher/Analysis/mr-to-pct_tests/pretrained_net_final_20220825.pth")
```
Run notebook.

This will produce the output pCT image in the specified folder.


Feedback welcome at siti.yaakub@plymouth.ac.uk
