#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Classes, functions for Synaptic Filter
Created on Sun 17 Jan 2021
@author: jannes
"""

import numpy as np
import pandas as pd

from src.update_functions import *
from src.init_functions import *


def run_timeseries(p, hetero=False):
    # run pre post or correlated protocol, return variable changes

    v = init(p)
    if hetero==False:
        v = init_pre_post_protocol(v,p,wait=p['wait'])
        initial_readout_time = p['wait']
    else: # hetero True
        v, wait_STDP = init_correlated_protocol(v,p,wait=p['wait'])
        initial_readout_time = wait_STDP # after STDP is over

    # ensure positivity
    initial_readout_time = max(initial_readout_time,2*p['dt'])

    for k in range(p['t_num']-1):
        update_protocol(v,p,k)
        update_filter(v,p,k)

    # init and final state
    idx_0 = int(initial_readout_time/p['dt']) - 2
    idx_f = -2
    mu_0, sig2_0 = v['mu'][idx_0], v['sig2'][idx_0]
    mu_f, sig2_f = v['mu'][idx_f], v['sig2'][idx_f]

    return (v, (mu_0, sig2_0, mu_f, sig2_f))


def run_STDP(mp,p, verbose = True, hetero=False):

    out = []
    delta_Ts = np.linspace(mp['delta_T_min'],mp['delta_T_max'],mp['num_delta_T'])
    for j,delta_T in enumerate(delta_Ts):

        # update pars
        p_cur = p.copy()
        p_cur['delta_T'] = delta_T

        # run
        _, (mu_0, sig2_0, mu_f, sig2_f) = run_timeseries(p_cur,hetero=hetero)

        # save output
        out.append({'delta_T':delta_T,
                   'mu_0':mu_0,
                   'sig2_0':sig2_0,
                   'mu_f':mu_f,
                   'sig2_f':sig2_f})

        # progresss
        if verbose == True:
            print(np.ceil(100*j/len(delta_Ts)), end='\r')

    return(out)


def get_performance(v,p,k=0):

    # init
    MSE = np.nan
    #z = np.nan
    #z2 = np.nan
    L = np.nan
    L_pt = np.nan

    # MSE
    delta = v['mu'][k] - v['w'][k] # vec
    MSE = np.mean(delta**2) # num

    # moments (not implemented here)
    #vec = delta/v['sig2'][k] # vec
    #z, z2 = np.mean(vec**0.5), np.mean(vec)

    # likelihoods
    # catch instability
    gbar_gg_one = v['gbar'][k] > 1
    if gbar_gg_one is True:
        v['gbar'] = 0.99

    gmap_gg_one = v['gmap'][k] > 1
    if gmap_gg_one is True:
        v['gmap'] = 0.99

    L, L_pt = [np.log(v[gkey][k] / p['g0dt']) if v['y'][k] > 0 # spike
               else np.log((1 - v[gkey][k]) / (1 - p['g0dt'])) # no spike
               for gkey in ['gbar','gmap']]

    if p['compute_sig2']: # only for 1d compute sig2
        assert p['dim'] == 1
        sig2 = v['sig2'][k]
        return {'MSE':MSE, 'L':L, 'L_pt':L_pt, 'sig2':sig2,
                'gbar_gg_one':gbar_gg_one,
                'gmap_gg_one':gmap_gg_one}

    else:
        return {'MSE':MSE, 'L':L, 'L_pt':L_pt,
            'gbar_gg_one':gbar_gg_one,
            'gmap_gg_one':gmap_gg_one}


def shift_vars_back(v,k):
    v['gbar'][k + 1] = v['gbar'][k]
    v['g'][k + 1] = v['g'][k]
    v['y'][k + 1] = v['y'][k]
    v['gmap'][k + 1] = v['gmap'][k]
    for key in v.keys():
        v[key][0] = v[key][1]


def run_simulation(p, online=True, verbose = False, precomputed_generator=None):

    # prepare
    out = []
    t_stop = int(p['tau_ou']//p['dt']*p['epoch_num'])
    t_readout = int(p['tau_ou']*p['epoch_wait']//p['dt'])
    if online:
        t_num = 2
    else:
        p['t_num'] = t_stop + 1
        t_num = p['t_num']

    # run
    v = init(p,t_num = t_num)
    assert 'x_wiggle' in v

    # update w, Sx, g, y
    if precomputed_generator is not None:
        keys = ['Sx', 'y', 'w', 'g']
        for key in keys:
            v[key] = precomputed_generator[key]
        errors = {}
        print('loaded precomputed values for:',keys)

    for t in range(t_stop):
        k = 0 if online else t

        if precomputed_generator is None:
            errors = update_generator(v,p,k) # ground truth weights, spikes

        update_protocol(v,p,k)  # compute kernels
        update_filter(v,p,k)

        if online is True:
            shift_vars_back(v,k)

        if t >= t_readout:
            res = get_performance(v,p,k=k)
            res.update(errors) # include errors in output
            out.append(res)

        if ((verbose is True) and (t % int(p['tau_ou']//p['dt']) == 0)):
            # progresss
            print(np.ceil(100*t/t_stop), end='\r')

    return pd.DataFrame(out), v
