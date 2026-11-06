# CARNet: Convolutional Attention Residual Network for Rainfall Prediction

[![Paper](https://img.shields.io/badge/Paper-IEEE-00629B)](https://doi.org/10.1109/ICCIT68739.2025.11490463)
[![DOI](https://img.shields.io/badge/DOI-10.1109%2FICCIT68739.2025.11490463-blue)](https://doi.org/10.1109/ICCIT68739.2025.11490463)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Official implementation of the **Convolutional Attention Residual Network**, or **CARNet**, proposed for accurate and computationally efficient rainfall prediction.

CARNet combines one-dimensional convolution, multi-head self-attention, and residual learning to capture local rainfall patterns and long-range temporal dependencies without relying on recurrent processing.

## Paper

> **A Hybrid Convolutional Attention Residual Network (CARNet) Framework for Enhanced Rainfall Prediction**  
> Swaranjit Roy, Omon Das, Mir Tawsif Sultan, and Sajal Kumar Adhikary  
> 2025 28th International Conference on Computer and Information Technology (ICCIT)

**DOI:** [https://doi.org/10.1109/ICCIT68739.2025.11490463](https://doi.org/10.1109/ICCIT68739.2025.11490463)

## Abstract

Accurate rainfall prediction is important for water-resource management, agricultural planning, flood preparedness, and climate-risk assessment. Although Long Short-Term Memory, or LSTM, and Convolutional LSTM models can learn temporal and spatiotemporal rainfall dynamics, their sequential operations may require substantial training time and computational resources.

CARNet provides a recurrence-free alternative that integrates:

- Conv1D layers for local temporal feature extraction
- multi-head self-attention for long-range dependencies
- residual connections for stable gradient propagation
- dropout regularization to reduce overfitting
- feedforward processing for parallel computation

The framework was evaluated using monthly rainfall data from eight locations in Bangladesh. The study compared CARNet with LSTM and ConvLSTM using predictive accuracy, GPU usage, and training time.

CARNet achieved an overall coefficient of determination of 0.93, while reducing MAE by 27% and RMSE by 25% relative to ConvLSTM. Compared with LSTM, CARNet reduced reported GPU usage by 54–67% and training time by 72–74%.

## Model Overview

CARNet processes a multivariate rainfall sequence through the following stages:

1. Input time-series window
2. Initial Conv1D feature extraction
3. Four multi-head self-attention blocks
4. ReLU activation within each attention block
5. Residual feature updates
6. Final Conv1D layer
7. Dropout regularization
8. Dense rainfall-prediction output

Unlike LSTM and ConvLSTM, CARNet does not process the input one time step at a time. Its attention and convolution operations can be computed in parallel, reducing the sequential bottleneck associated with recurrent architectures.

## Architecture

```text
Input rainfall sequence
        |
        v
Initial Conv1D layer
        |
        v
Multi-Head Attention Block 1
        |
        v
ReLU activation
        |
        v
Residual addition
        |
        v
Multi-Head Attention Block 2
        |
        v
ReLU activation
        |
        v
Residual addition
        |
        v
Multi-Head Attention Block 3
        |
        v
ReLU activation
        |
        v
Residual addition
        |
        v
Multi-Head Attention Block 4
        |
        v
ReLU activation
        |
        v
Residual addition
        |
        v
Final Conv1D layer
        |
        v
Dropout
        |
        v
Dense output
```

## Attention Formulation

For query matrix \(Q\), key matrix \(K\), value matrix \(V\), and key dimension \(d_k\), scaled dot-product attention can be expressed as:

$$
\mathrm{Attention}(Q,K,V)
=
\mathrm{softmax}
\left(
\frac{QK^{T}}{\sqrt{d_k}}
\right)V
$$

Multi-head attention applies attention in multiple learned representation subspaces:

$$
\mathrm{MultiHead}(Q,K,V)
=
\mathrm{Concat}
\left(
\mathrm{head}_1,
\ldots,
\mathrm{head}_h
\right)W^O
$$

where each attention head is defined as:

$$
\mathrm{head}_i
=
\mathrm{Attention}
\left(
QW_i^Q,
KW_i^K,
VW_i^V
\right)
$$

A residual attention block can be represented conceptually as:

$$
X_{l+1}
=
X_l+
\mathrm{ReLU}
\left(
\mathrm{MultiHead}(X_l,X_l,X_l)
\right)
$$

The residual connection preserves the block input while allowing the attention mechanism to learn an additional feature refinement.

## Dataset

The study used monthly rainfall observations from 1990 to 2019 across eight locations in Bangladesh:

- Chittagong
- Dhaka
- Khepupara
- Rajshahi
- Rangamati
- Rangpur
- Sreemangal
- Sylhet

The input variables were assembled from the following sources:

- ground-based rainfall observations
- CHIRPS
- ERA5
- PERSIANN

Merging the sources produced 360 temporally aligned records before preprocessing.

## Data Preprocessing

The preprocessing pipeline consisted of the following steps:

1. Merge rainfall measurements from the available sources.
2. Check the time series for missing observations.
3. Apply a Yeo-Johnson transformation to stabilize variance and reduce skewness.
4. Detect and remove outliers using Isolation Forest.
5. Use an Isolation Forest contamination value of `0.15`.
6. Retain 304 cleaned observations after outlier removal.
7. Split the observations chronologically to avoid future-data leakage.
8. Construct samples using a sliding-window procedure.

Chronological splitting is essential for rainfall forecasting because a randomized split can move future observations into the training set and produce an unrealistically optimistic evaluation.

## Forecasting Window

A nine-month input window was used to predict rainfall three months ahead.

For example:

```text
Input:  January to September
Target: December
```

In general, given an input sequence ending at time \(t\), the forecast target is:

$$
\hat{y}_{t+3}
=
f
\left(
X_{t-8},
X_{t-7},
\ldots,
X_t
\right)
$$

where:

- \(X\) contains rainfall features from CHIRPS, ERA5, and PERSIANN
- \(f\) represents the forecasting model
- \(\hat{y}_{t+3}\) is the predicted rainfall three months after the input window

## Evaluation Metrics

The models were evaluated using the coefficient of determination, mean absolute error, and root mean squared error.

### Coefficient of determination

$$
R^2
=
1-
\frac{
\sum_{i=1}^{n}(y_i-\hat{y}_i)^2
}{
\sum_{i=1}^{n}(y_i-\bar{y})^2
}
$$

A value closer to 1 indicates that the model explains more of the observed rainfall variation.

### Mean absolute error

$$
\mathrm{MAE}
=
\frac{1}{n}
\sum_{i=1}^{n}
\left|y_i-\hat{y}_i\right|
$$

MAE measures the average absolute difference between observed and predicted rainfall.

### Root mean squared error

$$
\mathrm{RMSE}
=
\sqrt{
\frac{1}{n}
\sum_{i=1}^{n}
(y_i-\hat{y}_i)^2
}
$$

RMSE places greater emphasis on large prediction errors.

## Prediction Results

### Location-Level Performance

| Location | CARNet R² | ConvLSTM R² | LSTM R² | CARNet MAE | ConvLSTM MAE | LSTM MAE | CARNet RMSE | ConvLSTM RMSE | LSTM RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Chittagong | 0.92 | 0.85 | 0.79 | 0.23 | 0.31 | 0.36 | 0.28 | 0.38 | 0.45 |
| Dhaka | 0.92 | 0.89 | 0.87 | 0.23 | 0.26 | 0.28 | 0.29 | 0.34 | 0.38 |
| Khepupara | 0.94 | 0.86 | 0.77 | 0.18 | 0.27 | 0.36 | 0.23 | 0.35 | 0.45 |
| Rajshahi | 0.91 | 0.84 | 0.78 | 0.22 | 0.31 | 0.38 | 0.29 | 0.37 | 0.45 |
| Rangamati | 0.92 | 0.85 | 0.81 | 0.23 | 0.32 | 0.37 | 0.29 | 0.39 | 0.44 |
| Rangpur | 0.94 | 0.86 | 0.79 | 0.21 | 0.30 | 0.36 | 0.25 | 0.37 | 0.45 |
| Sreemangal | 0.92 | 0.85 | 0.77 | 0.22 | 0.29 | 0.37 | 0.27 | 0.37 | 0.46 |
| Sylhet | 0.94 | 0.89 | 0.83 | 0.19 | 0.25 | 0.31 | 0.23 | 0.32 | 0.40 |
| **Overall** | **0.93** | **0.86** | **0.80** | **0.21** | **0.29** | **0.35** | **0.27** | **0.36** | **0.43** |

CARNet obtained the highest R² and lowest MAE and RMSE at every evaluated location.

### Improvement over ConvLSTM

| Location | R² improvement | MAE reduction | RMSE reduction |
|---|---:|---:|---:|
| Chittagong | 8% | 26% | 26% |
| Dhaka | 3% | 12% | 15% |
| Khepupara | 9% | 33% | 34% |
| Rajshahi | 8% | 29% | 22% |
| Rangamati | 8% | 28% | 26% |
| Rangpur | 9% | 30% | 32% |
| Sreemangal | 8% | 24% | 27% |
| Sylhet | 6% | 24% | 28% |
| **Overall** | **8%** | **27%** | **25%** |

The largest reported RMSE reduction relative to ConvLSTM was 34% at Khepupara.

## Computational Efficiency

| Location | CARNet training time | ConvLSTM training time | LSTM training time | GPU reduction vs. LSTM | Time reduction vs. LSTM |
|---|---:|---:|---:|---:|---:|
| Chittagong | 132 | 133.67 s | 512.39 s | 514.85 s | 58% | 74% |
| Khepupara | 129.49 s | 499.03 s | 489.27 s | 61% | 74% |
| Rajshahi | 133.86 s | 518.48 s | 502.87 s | 65% | 74% |
| Rangamati | 127.63 s | 495.37 s | 484.36 s | 54% | 74% |
| Rangpur | 146.17 s | 483.18 s | 489.10 s | 61% | 70% |
| Sreemangal | 131.70 s | 490.11 s | 470.09 s | 59% | 72% |
| Sylhet | 129.64 s | 496.75 s | 476.44 s | 67% | 73% |

Across the eight locations, CARNet completed training in approximately 127–147 seconds. ConvLSTM required approximately 483–519 seconds, while LSTM required approximately 470–515 seconds.

The reported efficiency improvements arise from CARNet's recurrence-free architecture. Conv1D and attention operations support parallel processing, while LSTM and ConvLSTM depend on sequential state propagation.

## Key Findings

- CARNet achieved an overall R² of **0.93**.
- CARNet achieved an overall MAE of **0.21**.
- CARNet achieved an overall RMSE of **0.27**.
- R² values ranged from **0.91 to 0.94** across the eight stations.
- MAE values ranged from **0.18 to 0.27**.
- RMSE values ranged from **0.22 to 0.30**.
- CARNet improved R² by an average of **8%** relative to ConvLSTM.
- CARNet reduced MAE by an average of **27%** relative to ConvLSTM.
- CARNet reduced RMSE by an average of **25%** relative to ConvLSTM.
- CARNet reduced GPU usage by **54–67%** relative to LSTM.
- CARNet reduced training time by approximately **72–74%** relative to LSTM.

These results apply to the datasets, preprocessing decisions, architectures, and experimental conditions described in the paper. They should not be interpreted as a guarantee of superior performance on every rainfall dataset.

## Suggested Repository Structure

```text
carnet-rainfall-prediction/
├── README.md
├── LICENSE
├── carnet.py
├── preprocessing.py
├── train.py
├── evaluate.py
├── models/
│   ├── carnet.py
│   ├── lstm.py
│   └── convlstm.py
├── notebooks/
│   └── CARNet_experiments.ipynb
├── data/
│   └── README.md
└── results/
    ├── metrics.csv
    └── figures/
```

Large datasets and trained model files should not be committed directly unless their licenses and file sizes permit redistribution. Use `.gitignore` for local datasets, checkpoints, and generated artifacts.

## Reproducibility Guidelines

For a valid comparison between CARNet, ConvLSTM, and LSTM, keep the following components consistent:

- chronological training, validation, and testing periods
- input and target variables
- nine-month input window
- three-month prediction horizon
- normalization and inverse-transformation procedures
- outlier-removal procedure
- random seed
- optimizer and learning rate
- batch size
- number of training epochs
- early-stopping criteria
- hardware environment
- evaluation metrics

Only the model architecture should change during a controlled architecture comparison.

## Limitations

The current study has several limitations:

- evaluation was limited to eight stations in Bangladesh
- monthly data were used instead of daily or hourly observations
- only rainfall-derived remote-sensing predictors were considered
- model performance under substantially different climatic conditions requires further evaluation
- computational measurements can vary with hardware, software versions, batch size, and implementation details
- the study does not establish that CARNet will outperform recurrent models on every forecasting task

Future research may extend CARNet to additional climatic regions, finer temporal resolutions, and supplementary hydrometeorological variables such as temperature, humidity, wind speed, pressure, soil moisture, and atmospheric circulation indicators.

## Applications

Potential applications include:

- operational rainfall forecasting
- flood early-warning systems
- reservoir inflow planning
- agricultural water management
- drought and climate-service platforms
- rainfall-driven hydrological modeling
- resource-constrained forecasting systems

Operational deployment should include uncertainty analysis, continuous recalibration, missing-data handling, and monitoring for changes in the underlying rainfall distribution.

## Citation

If you use CARNet, the experimental methodology, or the reported results in your research, please cite the paper through its DOI:

[https://doi.org/10.1109/ICCIT68739.2025.11490463](https://doi.org/10.1109/ICCIT68739.2025.11490463)

```bibtex
@inproceedings{roy2025carnet,
  title     = {A Hybrid Convolutional Attention Residual Network (CARNet) Framework for Enhanced Rainfall Prediction},
  author    = {Roy, Swaranjit and Das, Omon and Sultan, Mir Tawsif and Adhikary, Sajal Kumar},
  booktitle = {2025 28th International Conference on Computer and Information Technology (ICCIT)},
  year      = {2025},
  publisher = {IEEE},
  doi       = {10.1109/ICCIT68739.2025.11490463},
  url       = {https://doi.org/10.1109/ICCIT68739.2025.11490463}
}
```

Before using the BibTeX entry in a publication, compare it with the official citation exported from the DOI or IEEE Xplore record.

## License

The source code in this repository may be released under the MIT License. Add an official `LICENSE` file before publishing the repository.

The research paper remains subject to the copyright and usage conditions of its publisher. The repository license applies only to code and other materials for which the repository authors hold the necessary rights.

## Acknowledgment

If this repository contributes to your research, please cite the original paper and link to this repository.
