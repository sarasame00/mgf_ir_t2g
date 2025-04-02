import numpy as np
from .ohmatrix import ohmatrix

def ohfit(sampling, M, **kwargs):
    '''
    Fits both scalar components of a matrix-like object M using a sampling strategy.

    This function assumes that M has a decomposition structure (e.g., M = a*I + b*V),
    where 'a' and 'b' are scalar-valued components suitable for IR basis projection
    or other sampling-based compression techniques.

    Parameters:
    - sampling (object): An object with a .fit() method (e.g., IR basis sampler)
    - M (object): Object with attributes M.a and M.b representing scalar components
    - **kwargs: Additional keyword arguments passed to the fit() method

    Returns:
    - ohmatrix: A reconstructed object with fitted components for M.a and M.b
    '''
    return ohmatrix(sampling.fit(M.a, **kwargs), sampling.fit(M.b, **kwargs))


def ohevaluate(sampling, M, **kwargs):
    '''
    Evaluates both scalar components of a matrix-like object M using a sampling strategy.

    Parameters:
    - sampling (object): An object with an .evaluate() method (e.g., IR basis evaluator)
    - M (object): Object with attributes M.a and M.b representing scalar components
    - **kwargs: Additional keyword arguments passed to the evaluate() method

    Returns:
    - ohmatrix: A reconstructed object with evaluated components for M.a and M.b
    '''
    return ohmatrix(sampling.evaluate(M.a, **kwargs), sampling.evaluate(M.b, **kwargs))


def fprint(string, file, **kwargs):
    '''
    Prints a string to both standard output and a file.

    Useful for logging messages during iterative solvers or convergence diagnostics,
    ensuring visibility in both console and persistent file output.

    Parameters:
    - string (str): The text to be printed
    - file (file-like object): Target file to write the string to
    - **kwargs: Additional keyword arguments passed to the print function (e.g., end, flush)
    '''
    print(string, **kwargs)
    print(string, file=file, **kwargs)

