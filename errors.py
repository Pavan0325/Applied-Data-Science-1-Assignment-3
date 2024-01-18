""" 
Module errors. Contains:
error_prop Calculates the error range caused by the uncertainty of the fit
    parameters. Covariances are taken into account.
cover_to_corr: Converts covariance matrix into correlation matrix.
"""


import numpy as np

def error_prop(x, func, parameters, covar):
    """
    Calculates 1-sigma error ranges for a number or array using error propagation.

    Args:
        x (array-like): Input values for the function.
        func (function): The function to evaluate.
        parameters (array-like): Parameter values for the function.
        covar (numpy.ndarray): Covariance matrix of the parameters.

    Returns:
        numpy.ndarray: 1-sigma error ranges for the function output.
    """

    sigma = np.zeros_like(x)  # Initialize error ranges
    
    for i in range(len(parameters)):
        deriv1 = deriv(x, func, parameters, i)
        sigma += deriv1**2 * covar[i]  # Accumulate variances using variance only

    sigma = np.sqrt(sigma)  # Calculate standard deviations

    return sigma


def deriv(x, func, parameters, ip):
    """
    Calculates numerical derivatives of a function with respect to a parameter.

    Args:
        x (array-like): Input values for the function.
        func (function): The function to evaluate.
        parameters (array-like): Parameter values for the function.
        ip (int): Index of the parameter to differentiate with respect to.

    Returns:
        numpy.ndarray: Numerical derivative of the function with respect to the parameter.
    """

    scale = 1e-6  # Scale factor for numerical differentiation
    delta = np.zeros_like(parameters, dtype=float)
    delta[ip] = scale * np.abs(parameters[ip])  # Perturb the parameter

    diff = 0.5 * (func(x, *parameters + delta) - func(x, *parameters - delta))
    dfdx = diff / delta[ip]  # Calculate the derivative

    return dfdx


def covar_to_corr(covar):
    """
    Converts a covariance matrix to a correlation matrix.

    Args:
        covar (numpy.ndarray): Covariance matrix.

    Returns:
        numpy.ndarray: Correlation matrix.
    """

    sigma = np.sqrt(np.diag(covar))  # Standard deviations
    corr = covar / np.outer(sigma, sigma)  # Correlation matrix

    return corr

                       
