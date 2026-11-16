FL-RGT2FNN: MATLAB Implementation
Overview:
This repository contains the MATLAB implementation of the Function-Link Recurrent General Type-2 Fuzzy Neural Network (FL-RGT2FNN) developed for nonlinear system identification.
The proposed network combines function-link expansion, recurrent fuzzy neural network architecture, general type-2 fuzzy sets, online structure learning, and Lyapunov-based parameter adaptation.
==================================
Requirements:
MATLAB: any version
Required MATLAB toolboxes: without any toolboxes.
Operating system: Windows
====================================
Repository Contents:
Example_1_Training/: Nonlinear system identification training
Example_1_Testing/: To test the nonlinear system and generate Fig. 4.
Example_2_Training/: Time varying system training
Example_2_Testing/: To test time varying system and generate Fig. 5.
Gas_Furnace_Testing/: To test Gas furnace system and generate Fig. 6.
tesdata.mat/: Testing data for Gas furnace system.
Parameterlearn_Proposed_2.mat/: Traning parameters for Gas furnace system
==============================
Running the Experiments:
Download or clone the repository.
Open MATLAB and navigate to the relevant experiment folder.
Run the main MATLAB script for the selected Example.
Compare the generated RMSE values and figures with those reported in the manuscript.
====================================
Reproducibility:
The experiment scripts provide the model initialization, parameter settings, training configurations, and evaluation procedures used in the paper.
=====================================
Dataset:
The Box–Jenkins gas furnace dataset is publicly available at:http://openmv.net/info/gas-furnace
Original reference:
Box, G. E. P., and Jenkins, G. M. (1976). Time Series Analysis: Forecasting and Control. Revised edition. Holden-Day.
======================================
Citation:
If you use this implementation, please cite the associated journal paper and the archived software release.
Journal paper: Nonlinear System Identification Using Function-Link Recurrent General Type-2 Fuzzy Neural Networks: A Comparative Study
Software DOI: [Insert Zenodo DOI]
=====================================
License:
This project is licensed under the MIT License.
See the [LICENSE](LICENSE) file for details.
