# Gaussian Network Inference
Author: Gurpinder

Last update: July 2024

A python implementation of network inference algorithm based on gaussian processes, slightly modified from [1]. This program applies Gaussian Process Regression (GPR) to infer relationships between gene expression levels. It uses the **GPflow** to construct GPR models and calculates log marginal likelihoods. Additionally, it supports Bayesian inference to quantify the strength of relationships between genes and provides functions for debugging and analysis.

The jupyter notebook ```shell GP_example_new.ipynb``` contains a working example, using the data in ```shell dummy_data_format.csv``` file. ```shell dummy_data_format.csv``` also provides the format that can be used as a direct input to the ```shell GP_example_new.ipynb``` code, if you'd like to directly use the code given. ```shell GRN_functions.py``` contains functions that do the backend calculations and are imported into the ```shell GP_example_new.ipynb```.





