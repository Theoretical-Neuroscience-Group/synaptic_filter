#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Synaptic Filter: notebook simulations.

Created on 2021-02-07
@author: jannes
"""

# =============================================================================
#  load libs
# =============================================================================
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import itertools as it
import sys

from src.plotting import *
from src.update_functions import *
from src.init_functions import *
from src.run_functions import *
from util.util import save_obj, load_obj

def assign_parameters_to_maximally_imax_nodes(i_max, i, parameter_list):
    num_pars = len(parameter_list)
    ii_to_run = np.where(np.arange(num_pars)[[int(ii/(num_pars/i_max)) for ii in range(num_pars)]] == i)[0]
    parameter_list_to_run = [parameter_list[ii] for ii in ii_to_run]
    return parameter_list_to_run

def expspace(a0, an, n=50):
    """ linspace in exp space """
    return (a0 * np.exp(np.log(an / a0) * np.linspace(0, 1, n)))

def main(**kwargs):
    for k, v in kwargs.items():
        print('keyword argument: {} = {}'.format(k, v))

    # argument
    i = int(kwargs['i'])
    i_max = int(kwargs['i_max']) # number of workers

    """
        copy past from jupytern notebook
    """

    # defaults

    p = {'t_num': 4000,
     'dt': 0.001,
     'dim':2,
     'tau':0.025,
     'g0':1,
     'beta':0.5,
     'mu_ou':0,
     'sig2_ou':1,
     'tau_ou':1000, # s
     'rule':'corr'}
    p['g0dt'] = p['g0']*p['dt']

    # STDP
    p['delta_T'] = 0.01
    p['wait'] = 0.5 # s

    # correlation protocol
    p['correlated_times'] = np.array([0, 0.01]) # s, two spikes

    # bias
    p['include-bias'] = False
    p['sig2_oub'] = 1
    p['tau_oub'] = 0.025
    p['mu_oub'] = 1.0
    # spike response
    p['include-spike-response-kernel'] = True
    p['tau_alpha'] = 0.025
    p['amplitude_alpha'] = -3/p['beta']
    # single vector rules
    p['tau_z'] = 1

    # performance sims:
    p['epoch_num'] = 10
    p['epoch_wait'] = 2
    p['rate'] = 40 # Hz

    if 1: # performance sims 3.3.2021

        p['tau_ou'] = 4
        p['tau_d'] = 4
        p['tau_x_wiggle'] = 6
        p['beta'] = 0.3
        p['dim'] = 100
        p['dt'] = 0.001
        p['epoch_num'] = 30
        p['include-spike-response-kernel'] = False
        p['include-bias'] = False
        p['compute_sig2'] = False # for plotting?
        p['gamma_equal_g0'] = True

        mp = {'betas': [p['beta']], #np.linspace(0.01,1,21),
              'rules':  ('corr','exp','exp-rm2'),
              'repeats': range(10),
              'dims': [1, 5, 10, 50, 100]}
        res = []

        parameter_list = list(enumerate(it.product(mp['dims'], mp['betas'],
                                                mp['rules'], mp['repeats'])))
        length = len(parameter_list)

        if i != -1:
            # maximally i_max sims
            if len(parameter_list) < i_max + 1:
                parameter_list = [parameter_list[i]]
            else:
                parameter_list = assign_parameters_to_maximally_imax_nodes(
                                i_max=i_max, i=i, parameter_list=parameter_list)


        for count, (dim, beta, rule, repeat) in parameter_list:
            print(count,'/',length,'rule:',rule,'beta:',beta, 'dim',dim)
            p['rule'] = rule
            p['dim'] = dim
            p['beta'] = beta/(dim**(0.5))

            out,v = run_simulation(p,verbose=True,online=False)

            out = out.mean().to_dict()
            out.update({'beta':beta,'rule':rule, 'dim':dim})
            out.update({'count': count})
            res.append(out)

        res = pd.DataFrame(res)

        if i != -1:
            save_obj(res, f'simtab{i}','./pkl_data/')
            print(f'finished {i}', '\n','--------------','\n')



if __name__=='__main__':
    main(**dict(arg.split('=') for arg in sys.argv[1:])) # kwargs
