import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

import gpflow
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd

# Tensorflow prints a lot of logging info -- this it just to quiet it down.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

np.random.seed(123)
tf.random.set_seed(123)

def get_likelihood(X : np.array, Y: np.array) -> tuple:
    """
    Helper function meant to be called by get_network(). 

    Initialises a Gaussian regression model with Matern 3/2 kernel. 
    Gets Maximum-likelihood estimates for parameters using scipy optimiser (BFGS by default).
    
    returns log marginal likelihood and gpflow model "m"
    """
    # X is the parent and Y is the target gene expression vector.
    k = gpflow.kernels.Matern32()
    m = gpflow.models.GPR(data=(X, Y), kernel=k, mean_function=None)
    opt = gpflow.optimizers.Scipy()
    _ = opt.minimize(m.training_loss, m.trainable_variables, options=dict(maxiter=100))

    return float(m.log_marginal_likelihood()), m


def plot_gp(parent, target, m, bayes_factors, x_label=None, y_label=None, save_file=False, plt_x_lim = (-0.1, 1.1)):
    """ 
    Helper function to be called by get_network()
    Samples 10 functions from the GPR model and plots it, for debugging? Visualization purposes?
    """
    ## generate test points for prediction
    xx = np.linspace(plt_x_lim[0], plt_x_lim[1], 100).reshape(100, 1)  # test points must be of shape (N, D)

    ## predict mean and variance of latent GP at test points
    mean, var = m.predict_f(xx)

    ## generate 10 samples from posterior
    tf.random.set_seed(1)  # for reproducibility
    samples = m.predict_f_samples(xx, 10)  # shape (10, 100, 1)

    ## plot
    f, ax = plt.subplots(figsize=(12, 6))
    ax.plot(parent, target, "kx", mew=2)
    ax.plot(xx, mean, "C0", lw=2)
    ax.fill_between(
        xx[:, 0],
        mean[:, 0] - 1.96 * np.sqrt(var[:, 0]),
        mean[:, 0] + 1.96 * np.sqrt(var[:, 0]),
        color="C0",
        alpha=0.2,
    )

    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    textstr = f"lml = {round(float(m.log_marginal_likelihood()), 3)}\nBayes_factor_av = {round(bayes_factors[0],2)}\nBayes_factor_max = {round(bayes_factors[1],2)}"
    ax.text(0.8, 0.95, textstr, transform=ax.transAxes, fontsize=14,
        verticalalignment='top', bbox=props)

    ax.plot(xx, samples[:, :, 0].numpy().T, "C0", linewidth=0.5)
    _ = plt.xlim(plt_x_lim[0], plt_x_lim[1])
    if x_label: ax.set_xlabel(x_label)
    if y_label: ax.set_ylabel(y_label)
    
    if save_file: 
        plt.savefig(save_file, dpi = 300)
        plt.cla()
        plt.clf()

def calculate_random_log_marginal_likelihood(number_of_timepoints : int, number_of_random_genes : int = 5) -> float:
    """
    return log marginal likelihood of n random genes for a specified number of timepoints
    """

    # generate random gene parent arrays       
    x = np.random.normal(10, 1.0, size= (number_of_random_genes, number_of_timepoints))    
    y = np.random.normal(10, 1.0, size= (number_of_random_genes, number_of_timepoints))

    # calculate log marginal likelihood for each considered gene and store them in the array
    log_marginal_likelihoods = np.zeros(shape = (number_of_random_genes,))
    for gene in range(number_of_random_genes):
        x[gene] = (x[gene] - x[gene].min()) / (x[gene].max() - x[gene].min()) #TODO: NEED TO FIND A BETTER WAY!!!!
        y[gene] = (y[gene] - y[gene].min()) / (y[gene].max() - y[gene].min())
        log_marginal_likelihoods[gene], _ = get_likelihood(x[gene].reshape(-1,1), y[gene].reshape(-1,1))
    
    # return random average marginal likelihood
    return log_marginal_likelihoods.mean()


def calculate_bayes_factor(log_marginal_likelihood : np.array, random_log_marginal_likelihood: float) -> float:
    """
    Calculate bayes factor for two log marginal likelihoods. 
    """
    return np.log10(np.exp(log_marginal_likelihood)/np.exp(random_log_marginal_likelihood))
    


def get_network(parent_target_dict: dict, plot: dict={}) -> pd.DataFrame:
    """
    Main function to perform model inference on a list of genes.

    parent_target_dict: a dict of form {"Parent-Target": [[parent_expression_vector], [target_expression_vector]]}
    plot: dict with "Parent-Target" pairs to plot GPR model as keys and file names as values. - optional!
    """
    network_dataframe = pd.DataFrame(columns = ["From", "To", "log_marginal_likelihood","lengthscale","variance", "variance_n"])

    for gene_pair in parent_target_dict:
        log_marginal_likelihood, model = get_likelihood(parent_target_dict[gene_pair][0].reshape(-1,1), 
                                                parent_target_dict[gene_pair][1].reshape(-1,1))

        length_scale = float(np.array(model.kernel.variance))
        variance = float(np.array(model.kernel.lengthscales))
        variance_n = float(np.array(model.likelihood.variance))

        network_dataframe = pd.concat([ network_dataframe, 
                                        pd.DataFrame([[gene_pair.split("-")[0], 
                                        gene_pair.split("-")[1], 
                                        log_marginal_likelihood, 
                                        length_scale, variance, variance_n]], 
                                        columns = ["From", "To", "log_marginal_likelihood",
                                        "lengthscale","variance", "variance_n"]   )  ])
        
        # plot the gaussian function samples the gene-target pair if specified in "plot"
        if gene_pair in plot.keys(): plot_gp(parent_target_dict[gene_pair][:,0], 
                                                parent_target_dict[gene_pair][:,1], 
                                                model,
                                                x_label=gene_pair.split("-")[0], 
                                                y_label=gene_pair.split("-")[1], 
                                                save_file=plot[gene_pair])
    
    return network_dataframe


class experimental_functions():
    def get_likelihood_modified(X,Y):

        k = gpflow.kernels.Matern32()
        m = gpflow.models.GPR(data=(X, Y), kernel=k, mean_function=None)


        Variance_noise_bounds = (gpflow.utilities.to_default_float(0.01), gpflow.utilities.to_default_float(1))
        m.likelihood.variance = gpflow.Parameter(np.random.uniform(Variance_noise_bounds[0], Variance_noise_bounds[1]), transform=tfp.bijectors.Sigmoid(Variance_noise_bounds[0], Variance_noise_bounds[1]))

        global_lengthscale_bounds = (gpflow.utilities.to_default_float(0.2), gpflow.utilities.to_default_float(0.4))
        parent_expression_variance = np.var(X)
        #lengthscale_bounds = ( gpflow.utilities.to_default_float((global_lengthscale_bounds[0]/parent_expression_variance)-1e-3), 
        #                        gpflow.utilities.to_default_float((global_lengthscale_bounds[1]/parent_expression_variance)+1e-3))
        lengthscale_bounds = global_lengthscale_bounds
        m.kernel.lengthscales = gpflow.Parameter(np.random.uniform(lengthscale_bounds[0], lengthscale_bounds[1]), transform = tfp.bijectors.Sigmoid(lengthscale_bounds[0], lengthscale_bounds[1]))


        global_variance_bounds = (gpflow.utilities.to_default_float(0.2), gpflow.utilities.to_default_float(0.4))
        target_expression_variance = np.var(Y)
        # variance_bounds = ( gpflow.utilities.to_default_float((global_variance_bounds[0]/target_expression_variance)-1e-3), 
        #                         gpflow.utilities.to_default_float((global_variance_bounds[1]/target_expression_variance)+1e-3))

        variance_bounds = global_variance_bounds
        m.kernel.variance = gpflow.Parameter(np.random.uniform(variance_bounds[0], variance_bounds[1]), transform = tfp.bijectors.Sigmoid(variance_bounds[0], variance_bounds[1]))


        m.kernel.lengthscales.prior = tfd.Gamma(
            gpflow.utilities.to_default_float(10.0), gpflow.utilities.to_default_float(1.0)
        )

        m.kernel.variance.prior = tfd.Gamma(
            gpflow.utilities.to_default_float(10.0), gpflow.utilities.to_default_float(1.0)
        )

        opt = gpflow.optimizers.Scipy()
        opt_logs = opt.minimize(m.training_loss, m.trainable_variables, options=dict(maxiter=100))

        return float(m.log_marginal_likelihood()), m

    def get_likelihood_rational_quad(X, Y):
        """
        Helper function meant to be called by get_network(). 
        Initialises a Gaussian regression model with Matern 3/2 kernel and fits it using scipy optimiser (BFGS by default).
        returns log marginal likelihood and gpflow model "m"
        """
        # X is the parent and Y is the target gene expression vector.
        k = gpflow.kernels.RationalQuadratic()
        m = gpflow.models.GPR(data=(X, Y), kernel=k, mean_function=None)
        opt = gpflow.optimizers.Scipy()
        opt_logs = opt.minimize(m.training_loss, m.trainable_variables, options=dict(maxiter=100))

        return float(m.log_marginal_likelihood()), m

    def get_network_modified(parent_target_dict, plot={}):
        """
        Main function to perform model inference on a list of genes.

        parent_target_dict: a dict of form {"Parent-Target": [[parent_expression_vector], [target_expression_vector]]}
        plot: dict with "Parent-Target" pairs to plot GPR model as keys and file names as values. - optional!
        """
        network_dataframe = pd.DataFrame(columns = ["From", "To", "weight"])

        for i, gene_pair in enumerate(parent_target_dict):
            log_marginal_likeihood, model = experimental_functions.get_likelihood_rational_quad(parent_target_dict[gene_pair][:,0].reshape(-1,1), 
                                                    parent_target_dict[gene_pair][:,1].reshape(-1,1))

            network_dataframe = pd.concat([network_dataframe, 
                                        pd.DataFrame([[gene_pair.split("-")[0], gene_pair.split("-")[1], log_marginal_likeihood]],
                                                    columns = ["From", "To", "weight"])])
            
            # plot the gaussian function samples the gene-target pair if specified in "plot"
            if gene_pair in plot.keys(): plot_gp(parent_target_dict[gene_pair][:,0], parent_target_dict[gene_pair][:,1], model, x_label=gene_pair.split("-")[0], y_label=gene_pair.split("-")[1], save_file=plot[gene_pair])
        
        return network_dataframe    

    def fit_guassian_on_gene(Time, gene):
        k = gpflow.kernels.RationalQuadratic()
        m = gpflow.models.GPR((Time, gene), kernel=k, mean_function=None)

        opt = gpflow.optimizers.Scipy()
        opt_logs = opt.minimize(m.training_loss, m.trainable_variables, options=dict(maxiter=1000))
        return float(m.log_marginal_likelihood()), m

    def calculate_random_log_marginal_likelihood_deprecated(parent_data,target_data, number_of_timepoints : int, number_of_random_genes : int = 5) -> float:
        """
        return log marginal likelihood of n random genes for a specified number of timepoints
        """

        # generate random gene parent arrays       
        x = np.random.normal(loc = parent_data.mean(), scale = np.sqrt(np.var(parent_data)), size= (number_of_random_genes, number_of_timepoints))    
        y = np.random.normal(loc = target_data.mean(), scale = np.sqrt(np.var(target_data)), size= (number_of_random_genes, number_of_timepoints))

        # calculate log marginal likelihood for each considered gene and store them in the array
        log_marginal_likelihoods = np.zeros(shape = (number_of_random_genes,))
        for gene in range(number_of_random_genes):
            x[gene] = (x[gene] - x[gene].min()) / (x[gene].max() - x[gene].min()) #TODO: NEED TO FIND A BETTER WAY!!!!
            y[gene] = (y[gene] - y[gene].min()) / (y[gene].max() - y[gene].min())
            log_marginal_likelihoods[gene], _ = get_likelihood(x[gene].reshape(-1,1), y[gene].reshape(-1,1))
        
        # return random average marginal likelihood
        return log_marginal_likelihoods.mean()




