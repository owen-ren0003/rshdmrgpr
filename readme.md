# Introduction

This package contains the code that supplements the paper:  
  
Random Sampling High Dimensional Model Representation Gaussian Process Regression (RS-HDMR-GPR) for representing 
multidimensional functions with machine-learned lower-dimensional terms allowing insight with a general method

A supplementary notebook is provided in this package that the user can run to check the examples 
from the paper. Everything is coded up from start to finish, the user just need to run the code with respect
to a valid IPython kernel that has the following packages installed.

## Requirements
The following Python packages are required in order to use rshdmrgpr.

[matplotlib>=3.1.3](https://matplotlib.org/)  
[numpy>=1.18.1](https://numpy.org/)    
[pandas>=1.1.0](https://pandas.pydata.org/)  
[scikit-learn>=0.22.1](https://scikit-learn.org/stable/)  
[pytest>=6.2.5](https://docs.pytest.org/en/6.2.x/)  

# Installation

To install and use this package, have git installed and execute the following line within an anaconda environment command prompt:

```
pip install git+https://github.com/owen-ren0003/rshdmrgpr.git
```

```
import rshdmrgpr
```

Also to run the unittest, navigate to the rshdmrgpr directory and just type
```
pytest
```

Please consult the manual for more details.