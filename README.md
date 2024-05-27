# Generation of 3-component ground motion data by styleGAN2

## Operating Environment
The codes in this repository has been tested and is known to run under the following environment:
- Ubuntu 20.04.6 LTS
- conda 22.9.0
- Python 3.8.13
- pytorch 1.13.0
- numpy 1.23.3
- pandas 1.4.4
- matplotlib 3.5.3

## Usage
- The code for model definition and training is all included in the `main.py` file.
- A dataset must be prepared in advance and `input_file.csv` needs to be placed inside the `data` directory before training.

### Example structure of `input_file.csv`
For this code, it is necessary to prepare `input_file.csv` for loading ground motion data.
The meanings of the file headers are as follows.

| Header       | Description                                                   |
| ------------ | --------------------------------------------------            |
| `file_name`  | Path to the npy file containing a single ground motion data   |
| `mw`         | Moment magnitude, $M_W$                                       |
| `log_fault_dist` | Natural logarithm of rupture distance, $R_{\mathrm{RUP}}$            |
| `log10_pga`  | Common logarithm of the Peak Ground Acceleration (PGA) of ground motion |
| `log_v30`   | Natural logarithm of $V_{\mathrm{S}30}$                                   |
| `log_z1500`   | Natural logarithm of $Z_{1500}$                                           |

Each npy file contains a single ground motion data, and the values in each row correspond to the values of the condition labels.
The ground motion data must be normalized in amplitude.

### Structure of `example_*.npy` files
The example_*.npy files contain 2D arrays of ground motion time history data. The shape of the data is (data length x 3).

## License
This code is licensed under the MIT License.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
