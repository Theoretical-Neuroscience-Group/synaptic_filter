#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Classes, functions for Synaptic Filter
Created on Sun 17 Jan 2021
@author: jannes
"""

import numpy as np
from matplotlib import pyplot as plt

def rainbow(arr):
    """ return an array of colors used for plotting with length arr  """
    if type(arr)==int:
        return(plt.cm.jet(np.linspace(0,1,arr)))
    else:
        return(plt.cm.jet(np.linspace(0,1,len(arr))))

def plt_errorbar(xplt,yplt,yerr,label=None,lw=1,c='k',
                 marker='o',alpha=0.3,ls=None,color=None):
    """ same as plt.errorbar but with shaded area """
    cc = c if color is None else color
    if len(yerr.shape) > 1 and yerr.shape[0] == 2:  # two types of errors
        yerr_l, yerr_h = yerr[0], yerr[1]
    else:
        yerr_l, yerr_h = yerr, yerr
    #plt.plot(xplt, yplt, lw=lw, c=cc, marker=marker, ls=ls, label=label)
    plt.fill_between(xplt, yplt - yerr_l, yplt + yerr_h, color=cc, alpha=alpha)

def vplt(v,p,key='mu',dim=None,cut=1,c='k',alpha=1,return_xy=False,err=False):

    if key == 'filter' and dim is not None:
        tspan, yplt = vplt(v,p,key='mu',cut=cut,dim=dim,c='red',alpha=alpha,return_xy=True)
        vplt(v,p,key='w',cut=cut,c='k',alpha=alpha,dim=dim)
        # errorbars:
        if err is True:
            yerr = v['sig2'][:-cut,dim] if len(v['sig2'].shape)==2 else v['sig2'][:-cut,dim,dim]
            #if p['rule'] == 'exp-oja':

            plt.fill_between(tspan, yplt - yerr, yplt + yerr, color='red', alpha=0.3)
        return 0

    if len(v[key].shape) == 2:
        yplt = v[key][:-cut,dim]

    elif len(v[key].shape) == 1:
        yplt = v[key][:-cut]

    else: # a matrix, reshape
        yplt = v[key].reshape(len(v[key]),-1)[:-cut,dim]

    yplt = yplt.squeeze()
    tspan = np.arange(len(v[key])-cut)*p['dt']

    plt.plot(tspan, yplt ,label=key,lw=1,c=c,alpha=alpha)

    plt.ylabel(key)
    plt.xlabel('time [s]')

    if return_xy is True:
        return tspan, yplt
