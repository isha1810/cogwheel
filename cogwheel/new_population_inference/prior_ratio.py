from abc import ABC, abstractmethod
import pandas as pd
import numpy as np

class PriorRatio():
    numerator
    denominator
    params
    hyperparams

    @abstractmethod
    def lnprior_ratio(*params, *hyperparams):
        raise NotImplementedError

class RefPopulationtoIASPrior(PriorRatio):
    numerator = 'InjectionPrior'
    denominator = 'IASPrior'
    params = []
    hyperparams=[]

# class Population_to_pe_ratio(PriorRatio):
#     numerator = 'InjectionPrior'
#     denominator = 'IASPrior'
#     params = []
#     hyperparams=[]

class 


