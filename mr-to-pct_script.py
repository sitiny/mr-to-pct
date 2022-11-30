#!/usr/bin/env python
# mr-to-pct_script.py
# Runs mr to pct as a python script
# Inputs:
#   input_mr_file       Filename of your T1w MRI including full path
#   output_pct_file     Filename to give the output pCT including full path
#
# SNY: Wed 23 Nov 08:51:11 GMT 2022

import sys, os, torch
sys.path.append('../')
from utils.infer_funcs import do_mr_to_pct

# set device, use cuda if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set input and output data file paths
input_mr_file = sys.argv[1]
output_pct_file = sys.argv[2]

# Set trained model to load
saved_model = torch.load("pretrained_net_final_20220825.pth", map_location=device)

# Do you want to prepare the t1 image? This will perform bias correction and create a head mask
# yes = True, no = False. Output will be saved to _prep.nii
prep_t1 = True

# Do you want to produce an example plot? yes = True, no = False.
plot_mrct = False

print('T1w MR input file path: {}'.format(input_mr_file))
print('pCT output file path: {}'.format(output_pct_file))

# Run MR to pCT
do_mr_to_pct(input_mr_file, output_pct_file, saved_model, device, prep_t1, plot_mrct)
