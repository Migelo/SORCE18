#!/bin/bash

python tf.py spectra.npy Stroemgren_y.txt transmitted
python spectra_transmitted_and_opacity.py
