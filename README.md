# Flag n' Flare: Fast Linearly-Coupled Adaptive Gradient Methods
This repository contains the code to reproduce the results in [FLAG n' FLARE: Fast Linearly-Coupled Adaptive Gradient Methods](http://proceedings.mlr.press/v84/cheng18b/cheng18b.pdf).
FLAG and FLARE, described in the above paper, are two adaptive and accelerated optimization algorithms. 
I was fortunate to work on this project as an undergraduate at UC Berkeley in collaboration with [Xiang Cheng](https://www.linkedin.com/in/xiang-cheng-9818a63b/), [Fred Roosta](https://people.smp.uq.edu.au/FredRoosta/), [Peter Bartlett](https://www.stat.berkeley.edu/~bartlett/), and [Michael Mahoney](https://www.stat.berkeley.edu/~mmahoney/)

# Downloading the Data
The data can be found [here](https://drive.google.com/file/d/1NEm69sdVylKo8zZDuTFrtWh988zKGLQb/view?usp=sharing). 
Please download and unzip data.zip and place the resulting directory data/ in this repository's root.

# Running the Code
After downloading the data and placing it in the proper location, the code can be run from mainDriver.m. 
mainDriver.m allows the user to select the data to use, the type of regularization to apply, the initialization vector, and whether to measure performance versus number of iterations or number of proximal operator evaluations.
Each of these fields can be specified by uncommenting one of the options in the field's respective code block.
Depending on the data selected, the code will either run least squares regression or softmax classification (with the selected regularization).
As detailed in the paper, for the algorithms considered, performance with respect to the total number of proximal operator evaluations is more reflective of real world performance than performance versus iterations.
An expanded discussion on this point can be found in the paper at the beginning of Section 4 Numerical experiments.
For this reason, the code can be configured to display performance verses either the number of proximal operator evaluations or iterations.
After all optimizer finish, plots of the results will be displayed and results will be written to the results directory.
