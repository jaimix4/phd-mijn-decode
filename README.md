# phd-mijn-decode

This repository contains the code used/developed during my(mijn) PhD in plasma edge/SOL turbulent transport modeling using 2D gyrokinetic simulations with the code Gkeyll (https://github.com/ammarhakim/gkeyll). 

As for now each folder contains a different small project. 

## diff-op 

Script to quickly reproduce the results from the paper: A kinetic line-driven radiation operator and its application to Gyrokinetics, Jonathan Roeltgen etal 2025 Nucl.Fusion 65 106020.

In specific, the script `optimizer_eq12.py` finds the coefficients through minimizing the function defined in eq. 12 of the paper. This is done for Hydrogren case. This is a simple example for demonstration purposes. 

## roeltgen-opt-py

Repository to find the optimal coefficients for the kinetic line-driven radiation operator from paper:  A kinetic line-driven radiation operator and its application to Gyrokinetics, Jonathan Roeltgen etal 2025 Nucl.Fusion 65 106020. 

This repository intends to be a Python translation of the original MATLAB code used in the paper.

TO-DO: 
- Fix the optimizer to meet the performance from the MATLAB one (currently using: https://github.com/mechmotum/cyipopt).
- Extend the code to other species (He, C, Fe, etc) and other ionization states.
- Add a script to plot the results.

To run the optimizer: first install the dependencies (see `requirements.txt` or `environment.yml`), then run `python fit_manager.py`. 





