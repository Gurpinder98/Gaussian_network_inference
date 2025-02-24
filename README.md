# Gaussian Network Inference
Author: Gurpinder

Last update: July 2024

A python implementation of network inference algorithm based on gaussian processes, slightly modified from [1]. This program applies Gaussian Process Regression (GPR) to infer relationships between gene expression levels. It uses the **GPflow** to construct GPR models and calculates log marginal likelihoods. Additionally, it supports Bayesian inference to quantify the strength of relationships between genes and provides functions for debugging and analysis.

The jupyter notebook ```GP_example_new.ipynb``` contains a working example, using the data in ```dummy_data_format.csv``` file. ```shell dummy_data_format.csv``` also provides the format that can be used as a direct input to the ```GP_example_new.ipynb``` code, if you'd like to directly use the code given. ```GRN_functions.py``` contains functions that do the backend calculations and are imported into the ```GP_example_new.ipynb```.

## Requirements

This code has been tested to work with:
- python 3.10.12
- gpflow 2.5.2
- tensorflow 2.11.1
- pandas 2.0.3
- numpy 1.25.1

## Documentation

The minimal usage example is in ```GP_example_new.ipynb```, using ```dummy_data_format.csv```.

### ```GRN_functions.py``` functions

#### 1. ```get_likelihood(X: np.array, Y: np.array) -> tuple```
  Initializes a Gaussian regression model using a Matern 3/2 kernel and optimizes parameters using BFGS. Returns the log marginal likelihood and the trained model.

  **Parameters**:\
  ```X```: Parent gene expression vector (numpy array)\
  ```Y```: Target gene expression vector (numpy array)\
  **Returns**:\
  ```Log marginal likelihood``` (float)\
  ```Trained gpflow model```

#### 2. ```plot_gp(parent, target, m, bayes_factors, x_label=None, y_label=None, save_file=False, plt_x_lim=(-0.1, 1.1))```

  Visualizes GPR model predictions by sampling functions and plotting them along with confidence intervals.
  
  **Parameters**:\
  ```parent```: Parent gene expression data\
  ```target```: Target gene expression data\
  ```m```: Trained GPR model\
  ```bayes_factors```: List of Bayes factors\
  ```x_label```: Label for X-axis (optional)\
  ```y_label```: Label for Y-axis (optional)\
  ```save_file```: Path to save the plot (optional)\
  ```plt_x_lim```: X-axis limits (default: (-0.1, 1.1))
  
  **Output**:\
  Displays or saves the plot.


#### 3. ```calculate_random_log_marginal_likelihood(number_of_timepoints: int, number_of_random_genes: int = 5) -> float```

Computes log marginal likelihood for randomly generated gene expression values, used as a baseline for Bayesian inference.

**Parameters**:\
```number_of_timepoints```: Number of time points in the dataset\
```number_of_random_genes```: Number of random genes to generate (default: 5)

**Returns**:\
```Mean log marginal likelihood``` (float)


#### 4. ```calculate_bayes_factor(log_marginal_likelihood: np.array, random_log_marginal_likelihood: float) -> float```

Computes the Bayes Factor, quantifying the strength of evidence for the given gene relationship.

**Parameters**:\
```log_marginal_likelihood```: Log marginal likelihood of the observed data\
```random_log_marginal_likelihood```: Log marginal likelihood of random genes

**Returns**:\
```Bayes Factor``` (logarithmic scale)

#### 5. ```get_network(parent_target_dict: dict, plot: dict={}) -> pd.DataFrame```

Computes the log marginal likelihood for multiple gene pairs and returns a structured network representation in a DataFrame.

**Parameters**:\
```parent_target_dict```: Dictionary with key format "Parent-Target", mapping to a list of expression vectors\
```plot```: Dictionary specifying which gene pairs to visualize (optional)

**Returns**:\
```Pandas DataFrame``` with columns:\
*From*: Parent gene\
*To*: Target gene\
*log_marginal_likelihood*: Model likelihood estimate\
*lengthscale, variance, variance_n*: Model hyperparameters

#### 6. ```experimental_functions``` class

Contains alternative methods for model fitting and inference,\
```get_likelihood_modified(X, Y)```: Introduces parameter bounds and priors for the GPR model.\
```get_likelihood_rational_quad(X, Y)```: Uses a Rational Quadratic kernel instead of Matern 3/2.\
```get_network_modified(parent_target_dict, plot={})```: Alternative network inference method.\
```fit_gaussian_on_gene(Time, gene)```: Fits a GPR model to a single gene over time.\
```calculate_random_log_marginal_likelihood_deprecated(...)```: Older version of random likelihood calculation.


## Reference
1. Aijö, T., & Lähdesmäki, H. (2009). Learning gene regulatory networks from gene expression measurements using non-parametric molecular kinetics.\
Bioinformatics (Oxford, England), 25(22), 2937–2944. https://doi.org/10.1093/bioinformatics/btp511




