#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Classes, functions for Synaptic Filter
Created on Wed Aug 13 07:21 2020
@author: jannes
"""

# =============================================================================
#  load packages
# =============================================================================
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
from matplotlib import patches
#matplotlib.rcParams.update({'figure.figsize': (10,6)})
import itertools as it
import pandas as pd
from time import time
from time import sleep
from datetime import datetime
import scipy.stats
import argparse
import pickle
from scipy.linalg import sqrtm
matplotlib.rcParams.update({'font.size': 15})
from scipy.stats import multivariate_normal as mnorm
import scipy.special as ss
import os
from scipy.optimize import fmin
from collections import OrderedDict

class Fit(object):
    """ fit a dataset with arbitrary function in linear or log space """
    def __init__(self, x, y, logx=False, logy=False):
        self.logx = logx
        self.logy = logy
        self.x = np.log(x) if logx else x
        self.y = np.log(y) if logy else y

    def fit(self, fct, th0, maxiter=3000):
        """ min quadratic dist over dataset """
        self.fct = fct
        cost = lambda th: np.sum((self.fct(self.x, th) - self.y)**2)
        self.th = fmin(cost, th0, maxiter=maxiter)

    def plt(self,
            c='k',
            lw=None,
            label='Theory',
            alpha=1,
            marker=None,
            do_plot=True):
        """ plt fit on linear axis, if plt==False load output only"""
        if do_plot:
            xlim = plt.gca().get_xlim()
        else:
            xlim = np.array([np.min(self.x), np.max(self.x)])
            xlim = np.exp(xlim) if self.logx else xlim
        self.xlin = np.linspace(*xlim)
        flin = self.fct(np.log(self.xlin) if self.logx else self.xlin, self.th)
        self.ylin = np.exp(flin) if self.logy else flin
        if do_plot:
            plt.plot(self.xlin,
                     self.ylin,
                     c=c,
                     label=label,
                     lw=lw,
                     alpha=alpha,
                     marker=None)


def plt_legend(loc=None, ncol=1, prop={}, order=None, text_size=None):
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    labels, handles = list(by_label.values()), list(by_label.keys())
    # ordering
    if order is not None:
        if len(order) == len(labels):
            labels = list(np.array(labels)[order])
            handles = list(np.array(handles)[order])
        else:
            print('ignore order; mismatch with number of labels:', len(labels))
    if text_size is not None:
        prop.update({'size': text_size})
    plt.legend(labels, handles, loc=loc, ncol=ncol, prop=prop)


def expspace(a0, an, n=50):
    """ linspace in exp space """
    return (a0 * np.exp(np.log(an / a0) * np.linspace(0, 1, n)))


# functions
eta = np.random.randn
Phi = lambda x: ss.erf(x * 0.7071067811865475) * 0.5 + 0.5


# functions
def save_obj(obj, name, path='./'):
    if os.path.exists(path) == False:
        os.makedirs(path)
    with open(path + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name, path='./'):
    #    print('load obj:',os.getcwd())
    with open(path + name + '.pkl', 'rb') as f:
        return pickle.load(f)


def get_df(out_path, hetero=False):
    """ load pkl data frames and rm duplicates """
    print(
        'loading from:', out_path[out_path.find('c_'):]
        if out_path.find('c_') > 0 else out_path)
    tab_outs = []
    for file in os.listdir(out_path):
        if file.endswith(".pkl"):
            tab_out = load_obj(file[:-4], out_path)
            tab_outs.append(tab_out)

    #print('success rate:', len(tab_outs), '/', len(tab_out))
    out = pd.concat(tab_outs)  #.dropna()
    # drop non runs
    if hetero == False:
        out = out[np.isnan(out.beta) == False]
        out.loc[out.rule.apply(lambda s: ('sample' not in s) and
                               ('smooth' not in s)), 'tau_g'] = np.nan
        out = out.drop(columns=['tau_g'])
    out = out.sort_index()
    out['sim'] = out_path[-13:-5]
    out = out.drop_duplicates()
    
    # remove bad sims    
    #out = rm_sims(out,total_steps = 100*0.0001*)    
    return (out)


def folders2df(folders, basepath='./pkl_data/'):
    """ create dataframe from all .pkl files in folders (list) """
    out = pd.DataFrame([])
    for folder in folders:
        out = out.append(get_df(basepath + folder + '/'))
    return (out)


def add_key(tab, key, values):
    """ take data frame and augment """
    out = []
    for val in values:
        for i in range(len(tab)):
            mydict = dict(tab.iloc[i].items())
            mydict.update({key: val})
            out.append(mydict)
    out = pd.DataFrame(out)
    return (out)


def plt_errorbar(xplt,
                 yplt,
                 yerr,
                 label=None,
                 lw=2,
                 c='k',
                 marker='o',
                 alpha=0.3,
                 ls=None,
                 color=None):
    """ same as plt.errorbar but with shaded area """
    cc = c if color is None else color
    if len(yerr.shape) > 1 and yerr.shape[0] == 2:  # two types of errors
        yerr_l, yerr_h = yerr[0], yerr[1]
    else:
        yerr_l, yerr_h = yerr, yerr
    plt.plot(xplt, yplt, lw=lw, c=cc, marker=marker, ls=ls, label=label)
    plt.fill_between(xplt, yplt - yerr_l, yplt + yerr_h, color=cc, alpha=alpha)


class TimeSeries_AutoCorrelation(object):
    """ online computation of auto corrlation """
    def __init__(self, dim, L, store=False, downsample=1):
        """ call before loop: L is cutoff """

        # determine window to compute auto correlation
        self.L = L  #= int(fac*p['tau_ou']/p['dt'])
        self.dim = dim
        # down sample the correlation computation
        self.ds = downsample
        ds = downsample if downsample > 0 else 1
        #self.vars = [m for m in mets if '_std' not in m]
        # init
        self.mean = np.zeros(self.dim)
        self.corr = (np.zeros([L + 1, self.dim]))[::ds]

        #self.series = np.ones([2*L+1,self.dim])*np.nan
        self.count = -2 * self.L  # times called, start only after loading series

        # new
        self.kf = 0  # self.L # final index of relevant pointer zone
        self.series2 = np.ones([4 * L + 1, self.dim])  #*np.nan

        # averaging indices, back and forth
        self.i = (np.array(np.floor((np.arange(L + 1) + L) / 2),
                           dtype=np.int32))[::ds]
        self.j = (np.array(np.floor((np.arange(L + 1)[::-1]) / 2),
                           dtype=np.int32))[::ds]

        self.store = store
        if self.store:
            self.states = []

    def run_online(self, state):
        """ call during loop: feed in time series state(t) """

        # new:
        self.series2[self.kf] = state
        self.count += 1
        if self.count > 0:  # enter first time when kf = 2*L, count = 1
            self.mean += self.series2[self.kf - self.L]
            dk = self.kf - 2 * self.L
            if self.ds > 0:
                self.corr += self.series2[self.i + dk] * self.series2[self.j +
                                                                      dk]

        self.kf += 1
        # final position reached, copy 2nd half to first half of series2
        if self.kf == 4 * self.L:
            self.kf = 2 * self.L
            self.series2[:self.kf + 1] = self.series2[self.kf:]

    def post_process(self, include_sem=False):
        """ call after loop: output mean and sem of time series (2-d) """
        # normalise
        self.mean /= self.count
        if include_sem == False:
            return (self.mean)
        else:
            #self.corr[::self.ds] = self.corr[::self.ds]/self.count - self.mean**2
            self.corr = self.corr / self.count - self.mean**2
            # integrate over auto correlation, (-infty,inft) -> factor 2
            # and account for down sampling
            self.sem = 2 * self.ds * np.sum(self.corr, axis=0) / self.count
            self.sem = np.sign(self.sem) * np.abs(self.sem)**0.5  # sqrt
            return (np.hstack([self.mean, self.sem]))


class Variables(dict):
    """ Dict with methods """
    def __init__(self, *arg, **kw):
        # class acts like a dict
        super(Variables, self).__init__(*arg, **kw)

    def plt(self, key, dim=0, downSample=1):
        t_num = p['t_num']
        d = downSample
        if key in ('wm', 'lm'):
            m, s2 = ('m', 's2') if key == 'wm' else ('mu', 'sig2')
            tspan = (np.arange(0, t_num) * dt)[::d]
            plt.plot(tspan, self[m][::d, dim], 'b', label='theo' + str(dim))
            plt.plot(tspan, self[m][::d, dim] + self[s2][::d, dim]**0.5, ':b')
            plt.plot(tspan, self[m][::d, dim] - self[s2][::d, dim]**0.5, ':b')
            plt.plot(tspan, np.exp(
                self['lam'][::d, dim]), 'r') if key == 'wm' else plt.plot(
                    tspan, self['lam'][::d, dim], 'r')
        else:
            if len(self[key].shape) == 1:
                plt.plot(np.arange(0, t_num) * dt, self[key], label=key)
            else:
                plt.plot(np.arange(0, t_num) * dt,
                         self[key][:, dim],
                         label=key + str(dim))
            plt.gca().legend()
            plt.xlabel('time')
            plt.ylabel(key)

    def plt2(self, key, dim=0, downSample=1, lw=2, k_cut=0, n_sig=1):
        """ fancy version of plt
            n_sig : sigma environment represented by shaded area
        """
        d = downSample
        t_num = p['t_num']
        if key in ('lm', 'pf'):
            m, s2 = ('mu', 'sig2')
            tspan = (np.arange(0, t_num - 1 - k_cut) * dt)[::d]
            # ground truth
            yplt = self['lam'][k_cut:-1:d, dim]
            plt.plot(tspan, yplt, 'k', linewidth=lw)
            # filter
            yplt = self[m][k_cut:-1:d, dim]
            if len(self[s2][k_cut:-1:d].shape) > 1:
                err = n_sig * self[s2][k_cut:-1:d, dim, dim]**0.5
            else:
                err = n_sig * self[s2][k_cut:-1:d, dim]**0.5
            plt.fill_between(tspan,
                             yplt - err,
                             yplt + err,
                             alpha=0.3,
                             color='r')
            plt.plot(tspan, yplt, 'r', linewidth=lw)
        elif key == 'gs':
            tspan = (np.arange(0, t_num - 1 - k_cut) * dt)[::d]
            yplt = self['g'][k_cut:-1:d] / dt
            plt.plot(tspan, yplt, 'k', linewidth=lw, label='data')
            yplt = self['gbar'][k_cut:-1:d] / dt
            plt.plot(tspan,
                     yplt,
                     'r',
                     linewidth=lw,
                     label='prediction',
                     alpha=0.6)
            plt.xlabel('Time [s]')
            plt.ylabel('Firing rate [Hz]')
            plt_legend()

        elif key == 'mux':
            m, s2 = ('mu', 'sig2')
            tspan = (np.arange(0, t_num - 1 - k_cut) * dt)[::d]
            # ground truth
            yplt = self['x'][k_cut:-1:d, dim]
            plt.plot(tspan, yplt, 'k', linewidth=lw, alpha=0.5, label='x')
            # filter
            yplt = self[m][k_cut:-1:d, dim]
            # take diagonal if corr
            err = self[s2][k_cut:-1:d, dim]**0.5 if len(
                self[s2].shape) == 2 else self[s2][k_cut:-1:d, dim, dim]**0.5
            plt.fill_between(tspan,
                             yplt - err,
                             yplt + err,
                             alpha=0.3,
                             color='r')
            plt.plot(tspan, yplt, 'r', linewidth=lw, label='SF')
            plt.xlabel('Time [s]')
            plt.ylabel('Firing rate [Hz]')
            plt_legend()

        else:
            if len(self[key].shape) == 1:
                plt.plot(np.arange(0, t_num) * dt, self[key], label=key)
            else:
                plt.plot(np.arange(0, t_num) * dt,
                         self[key][:, dim],
                         label=key + str(dim))
            plt.gca().legend()
            plt.xlabel('Time [s]')
            plt.ylabel(key)

    def init(self, t_num, p):
        """ init variables """
        dim = int(p['dim'])
        dim_gm = int(p['dim-gm'])

        v = self
        v['lam'] = np.zeros((t_num, dim_gm))  # ground truth
        v['lam'][0] = np.ones(dim_gm) * p['mu_ou']
        v['mu'] = np.zeros((t_num, dim))
        if 'corr' in p['rule']:
            v['sig2'] = np.zeros((t_num, dim, dim))
            v['sig2'][0] = np.diag(np.ones(dim) * p['sig2_ou'])
        else:
            v['sig2'] = np.zeros((t_num, dim))
            v['sig2'][0] = np.ones(dim) * p['sig2_ou'] if np.isnan(
                p['lr']) else np.ones(dim) * p['lr']

        if 'init_deviation' not in p:
            p['init_deviation'] = np.ones(dim)
        v['mu'][0] = np.ones(dim) * p['mu_ou'] * p['init_deviation'][:dim]

        # weight space
        v['w'] = np.zeros((t_num, dim))  # sampled
        v['t'] = np.zeros(dim)
        v['x_ref'] = np.zeros(dim)

        # increase x artifically
        if dim_gm > dim:
            dd = dim_gm - dim
            v['t-gm'] = np.zeros(dd)
            v['x_ref-gm'] = np.zeros(dd)
            v['x-gm'] = np.zeros((t_num, dd))
        v['x'] = np.zeros((t_num, dim))
        v['g'] = np.ones(t_num) * p['g0'] * p['dt']
        v['u'] = np.zeros(t_num)
        v['gbar'] = np.ones(t_num) * p['g0'] * p['dt']
        v['gmap'] = np.ones(t_num) * p['g0'] * p[
            'dt']  # compute without variance
        v['y'] = np.zeros(t_num)
        v['y_ref'] = 0

        if 'sample' in p['rule']:
            v['sig2u'] = np.zeros(t_num)
            v['ubar'] = np.zeros(t_num)
            v['uw'] = np.zeros(t_num)  # sampling pot.
            v['sig2u'][0] = 1

        elif 'smooth' in p['rule']:
            v['sig2u'] = np.zeros(t_num)
            v['xy'] = np.zeros((t_num, dim))
            v['xg'] = np.zeros((t_num, dim))
            v['xxg'] = np.zeros(
                (t_num, dim, dim)) if 'corr' in p['rule'] else np.zeros(
                    (t_num, dim))
            v['uw'] = np.zeros(t_num)  # sampling pot.

        if 'pf' in p['rule']:
            v['W'] = np.ones((t_num, p['L'])) / p['L']  # particle weights
            v['p'] = np.zeros(
                (2, dim, p['L']))  # particle position (dont remember all)
            v['N'] = np.ones(t_num) * p['L']  # effective sample size
            v['p'][0, :, :] = p['mu_ou'] + p['sig2_ou']**0.5 * eta(dim, p['L'])

    def init_spikes(self, p):
        """ load spiking protocols (details specified in gen_tab)
            p['triplet']
            p['hetero-STDP']
            else: vanilla stdp
        """
        v = self
        # dT distance pre post
        # dS distance between pre spikes in burst
        # nS number of spikes in burst
        # nB number of bursts
        # dB distance between bursts
        T_wait = p['T_wait']  # = 1 # s
        dB = p['dB']  # [s]
        dT, dS, nS, nB = p['dT'], p['dS'], int(p['nS']), int(p['nB'])
        dT0 = np.abs(0 if np.isnan(dT) else dT)
        # burst distances (including final decay period), burst length and dT shift
        t_num = int((nB * (dB + nS * dS) + dT0 + 0. * T_wait) / p['dt'])
        p['t_num'] = t_num

        # init vars
        v.init(t_num, p)
        v['x'][:, 0] = p['bias']  #if p['hetero-STDP']==False else 0 # bias
        # load spike train
        v['Sx'] = np.zeros([t_num, p['dim']])
        # pre synaptic spikes per burst
        t_preB = np.arange(nS) * dS
        # copy each burst with increasing delay
        t_pre = np.sort(
            np.concatenate(
                [t_preB + iB * (dB + dS * nS) + dT0 for iB in range(nB)]))
        # add T_wait in the beginning
        t_pre += T_wait

        ### loading triplets ###
        if p['triplet']:
            # load pre and post spikes
            pre, post = ('Sx', 'y') if dT > 0 else ('y', 'Sx')
            v[pre][np.array(np.round(t_pre / p['dt']), dtype=np.int32)] = 1
            v[post][np.array(np.round((t_pre + np.abs(dT)) / p['dt']),
                             dtype=np.int32)] = 1

            # add 3rd spike with dT relative to single spike
            if np.isnan(p['dT_pre2']) == False and np.isnan(p['dT_post2']):
                idx2 = np.array(np.where(v['y'])[0] - p['dT_pre2'] / dt,
                                dtype=np.int32)
                v['Sx'][idx2] = 1
            elif np.isnan(p['dT_post2']) == False and np.isnan(p['dT_pre2']):
                idx2 = np.array(np.where(v['Sx'])[0] + p['dT_post2'] / dt,
                                dtype=np.int32)
                v['y'][idx2] = 1
            else:
                print('warning: loading triplet but no 3rd spike specified')

        ### loading heterosynaptic protocol ###
        elif p['hetero-STDP']:
            # load pre spikes
            pre, post = ('Sx', 'y') if dT > 0 else ('y', 'Sx')
            v[pre][np.array(np.round(t_pre / p['dt']), dtype=np.int32)] = 1
            v[post][np.array(np.round((t_pre + np.abs(dT)) / p['dt']),
                             dtype=np.int32)] = 1

            # PC protocol
            idx_max = 50
            if p['hetero-correlations']:
                idxs = np.arange(p['n_PC']) * 5 + 1 if 'n_PC' in p else [1]
                idxs = idxs + int(p['T_wait_PC'] / p['dt'])
                v['Sx'][idxs] = 1  # both dim
                idx_max = idxs.max()
            # knock out testing spikes
            if p['hetero-STDP-xSpikes'] == 'hetero':
                ihet = int(p['bias'])  # set first syn to zero
                v['Sx'][idx_max + 1:-1, ihet] = 0
            elif p['hetero-STDP-xSpikes'] == 'homo':
                ihom = int(p['bias']) + 1  # set 2nd syn to zero
                v['Sx'][idx_max + 1:-1, ihom] = 0
            #elif p['hetero-STDP-xSpikes'] == 'mixed':
            #   nothing to change
        ### loading STDP ###
        else:
            # load pre spikes
            if np.isnan(dT):
                v['Sx'][np.array(np.round(t_pre / p['dt']),
                                 dtype=np.int32)] = 1
            # load pre and post spikes
            else:
                pre, post = ('Sx', 'y') if dT > 0 else ('y', 'Sx')
                v[pre][np.array(np.round(t_pre / p['dt']), dtype=np.int32)] = 1
                v[post][np.array(np.round((t_pre + np.abs(dT)) / p['dt']),
                                 dtype=np.int32)] = 1


# =============================================================================
# Expo
# =============================================================================

    def exp(self, p, k):
        v = self
        # update with likelihood (w/o dt)
        if p['bayesian']:
            if 'smooth' in p['rule']:
                # represents all of the contributions of the bias
                v['sig2u'][k] = p['beta'] * (
                    0.5 * v['sig2'][k, 0] * p['beta'] +
                    v['mu'][k, 0]) if p['bias'] else 0
                v['gbar'][k] = p['g0dt'] * np.exp(v['uw'][k] * p['beta'] +
                                                  v['sig2u'][k])
                v['xy'][k + 1] = v['xy'][k] * (
                    1 -
                    dt / p['tau_s']) + v['y'][k] * v['x'][k] * dt / p['tau_s']
                v['xg'][k + 1] = v['xg'][k] * (1 - dt / p['tau_s']) + v[
                    'gbar'][k] * v['x'][k] * dt / p['tau_s']
                v['xxg'][k + 1] = v['xxg'][k] * (1 - dt / p['tau_s']) + (
                    v['gbar'][k] * v['x'][k]**2) * dt / p['tau_s']
                # plug together
                dmu_like = v['sig2'][k] * p['beta'] * (v['xy'][k] - v['xg'][k])
                dsig2_like = -(v['sig2'][k] * p['beta'])**2 * v['xxg'][k]
            else:
                term1 = v['sig2'][k] * v['x'][k] * p['beta']
                v['gmap'][k] = p['g0dt'] * np.exp(
                    v['mu'][k].dot(v['x'][k]) * p['beta'])
                v['gbar'][k] = v['gmap'][k] * np.exp(
                    v['sig2'][k].dot(v['x'][k]**2) * p['beta']**2 / 2)
                dmu_like = term1 * (v['y'][k] - v['gbar'][k])
                dsig2_like = -v['gbar'][k] * term1**2
            return (dmu_like, dsig2_like)
        # classical
        else:
            term1 = v['sig2'][k] * v['x'][k] * p['beta']
            v['gbar'][k] = p['g0dt'] * np.exp(
                v['mu'][k].dot(v['x'][k]) * p['beta'])
            dmu_like = term1 * (v['y'][k] - v['gbar'][k])
            dsig2_like = 0
            return (dmu_like, dsig2_like)

    def corr(self, p, k):
        v = self
        if 'smooth' in p['rule']:
            v['sig2u'][k] = p['beta'] * (0.5 * v['sig2'][k, 0, 0] * p['beta'] +
                                         v['mu'][k, 0]) if p['bias'] else 0
            v['gbar'][k] = p['g0dt'] * np.exp(v['uw'][k] * p['beta'] +
                                              v['sig2u'][k])
            v['xy'][k + 1] = v['xy'][k] * (
                1 - dt / p['tau_s']) + v['y'][k] * v['x'][k] * dt / p['tau_s']
            v['xg'][k + 1] = v['xg'][k] * (
                1 -
                dt / p['tau_s']) + v['gbar'][k] * v['x'][k] * dt / p['tau_s']
            v['xxg'][k + 1] = v['xxg'][k] * (1 - dt / p['tau_s']) + (
                v['gbar'][k] * v['x'][k][:, np.newaxis] *
                v['x'][k][np.newaxis, :]) * dt / p['tau_s']
            # plug together
            dmu_like = v['sig2'][k].dot(v['xy'][k] - v['xg'][k]) * p['beta']
            dsig2_like = -p['beta']**2 * v['sig2'][k].dot(v['xxg'][k].dot(
                v['sig2'][k]))
        else:
            term1 = v['sig2'][k].dot(v['x'][k]) * p['beta']
            v['gmap'][k] = p['g0dt'] * np.exp(
                v['mu'][k].dot(v['x'][k]) * p['beta'])
            v['gbar'][k] = v['gmap'][k] * np.exp(
                v['x'][k].dot(v['sig2'][k].dot(v['x'][k])) * p['beta']**2 / 2)
            dmu_like = term1 * (v['y'][k] - v['gbar'][k])
            dsig2_like = -v['gbar'][k] * term1[np.newaxis, :] * term1[:, np.
                                                                      newaxis]
        return (dmu_like, dsig2_like)

    def w_sample(self, k, ix, p):
        v = self
        if 'corr' in p['rule'] and len(ix) > 0:  # only sample if any ix
            v['w'][k, ix] = mnorm(v['mu'][k, ix],
                                  v['sig2'][k, ix][:, ix]).rvs()
        elif 'corr' not in p['rule']:  #exp
            v['w'][k, ix] = v['mu'][k, ix] + v['sig2'][k, ix]**0.5 * eta(
                len(ix))
        v['uw'][k +
                1] = v['uw'][k] * (1 - p['dt'] / p['tau']) + sum(v['w'][k, ix])
        if v['uw'][k + 1] == np.inf or np.isnan(v['uw'][k + 1]):
            print('fail in with bad uw sample', t)
            print('')
            []**2

    def pf(self, p, k):
        v = self
        # propagate hidden: prior (sampling from proposal distribution)
        v['p'][1] = v['p'][0] + dt / p['tau_ou'] * (
            p['mu_ou'] - v['p'][0]) + p['dW'] * eta(p['dim'], p['L'])
        # re-weight particle: likelihood (eq. 111)
        #active = (v['y_ref']<0 or v['y_ref']==p['t_ref'])
        active = True
        if active:
            g_i = p['g0'] * np.exp(p['beta'] * v['x'][k].dot(v['p'][0]))
            v['gbar'][k] = g_i.dot(v['W'][k])  # resampling at every k
            v['W'][k + 1] = (1 + (g_i - v['gbar'][k]) / v['gbar'][k] *
                             (v['y'][k] - v['gbar'][k] * dt)) * v['W'][k]
        else:  # refractory period
            v['gbar'][k] = 0
            v['W'][k + 1] = v['W'][k]

        # cut negative weights
        idx_negative = v['W'][k + 1] < 0
        if np.any(idx_negative):
            p['w<0'] += np.sum(idx_negative)
            v['W'][k + 1][idx_negative] = 0
            print('step', t, 'set negative weights to zero. W_t:',
                  v['W'][k + 1])

        # normalise and readout
        #v['W'][k+1][v['W'][k+1]<0] = 0 # set to zero
        v['W'][k + 1] = v['W'][k + 1] / v['W'][k + 1].sum()
        v['mu'][k + 1] = v['p'][1].dot(v['W'][k + 1])
        if p['rule'] == 'pf':
            v['sig2'][k + 1] = ((v['p'][1] -
                                 v['mu'][k + 1, :, np.newaxis])**2).dot(
                                     v['W'][1])
            # [dim] = (p[dim,L] - mu[dim,L])**2 dot W[L]
        elif 'corr' in p['rule']:
            deltas = v['p'][1] - v['mu'][k + 1, :, np.newaxis]  # [dim x L]
            v['sig2'][k + 1] = sum([
                w * delta[:, np.newaxis] * delta[np.newaxis, :]
                for w, delta in zip(v['W'][1], deltas.T)
            ])
        # effective particle number
        v['N'][k + 1] = 1 / np.sum(v['W'][k + 1]**2)
        # resample particles from categorical
        if v['N'][k + 1] < p['L'] * 0.75:
            #print(k+1,'resampling with N_eff:',v['N'][k+1])
            i = np.random.choice(p['L'], p['L'], p=v['W'][k + 1])
            v['p'][1] = v['p'][1, :, i].T
            v['W'][k + 1] = 1 / p['L']  # reset

    def get_prior(self, p, k):
        """
        compute prior contribution to weight change for rules and dims
        Used only for hetero
        """
        if p['bayesian'] == False:
            return (0, 0)

        # treat bias separaetly?
        bias = (p['bias'] and ('tau_oub' in p) and ('mu_oub' in p)
                and ('sig2_oub' in p))
        corr = 'corr' in p['rule']

        v = self

        if corr:
            if bias:
                dmu_pi = -(v['mu'][k] - p['mu_ou']) / p['tau_ou']
                dmu_pi[0] = -(v['mu'][k, 0] - p['mu_oub']) / p['tau_oub']

                # correlation decay with bias speed (1/tau_b + 1/tau_w approx 1/tau_b)
                # factor two removed for correlations
                dsig2_pi = -(v['sig2'][k] - np.diag(
                    np.ones(p['dim']) * p['sig2_oub'])) / p['tau_oub']
                # factor two added for bias
                dsig2_pi[0, 0] = 2 * dsig2_pi[0, 0]

                # weight decays slowly
                dsig2_pi[1:, 1:] = -2 * (v['sig2'][k, 1:, 1:] -
                                         p['sig2_ou']) / p['tau_ou']

            else:
                dmu_pi = -(v['mu'][k] - p['mu_ou']) / p['tau_ou']
                dsig2_pi = -2 * (v['sig2'][k] - np.diag(
                    np.ones(p['dim']) * p['sig2_ou'])) / p['tau_ou']
        else:  # diag
            if bias:
                dsig2_pi = -2 * (v['sig2'][k] - p['sig2_ou']) / p['tau_ou']
                dmu_pi = -(v['mu'][k] - p['mu_ou']) / p['tau_ou']
                # include bias pars
                dsig2_pi[0] = -2 * (v['sig2'][k, 0] -
                                    p['sig2_oub']) / p['tau_oub']
                dmu_pi[0] = -(v['mu'][k, 0] - p['mu_oub']) / p['tau_oub']
            else:
                dsig2_pi = -2 * (v['sig2'][k] - p['sig2_ou']) / p['tau_ou']
                dmu_pi = -(v['mu'][k] - p['mu_ou']) / p['tau_ou']

        return (dmu_pi, dsig2_pi)

    def res(self, p, k=0, end=None):  #,only_mean=False):
        """ compute performance measures
            p : parameters of the sim
            k, end : beginning and end index of time series
            output matches: 'mets' (metrics) specified in the main script

            currently implemented outputs:
                MSE: mean squared error
                p_in: fraction of time w* is in sigma range
                z: normalised mean estimator (=0 for perfect match)
                z2: normalised variance estimator (=1 for perfect match)
                z2d: loglikelihood wrt to baseline firing rate
                z2d_pt: MAP loglikelihood, only useful for (exp, corr) rules
                z2d0: LL of baseline g0
                z2dg: LL of true firing rate g(w*)
        """

        # init results output (ro):
        # mets: selection of metrics.
        # rule: for which rule computationa is carried out
        mets = p['mets']
        rule = p['rule']
        ro = list(np.ones([len(mets), 1]) * np.nan)  # result output
        v = self

        ### no model mismatch
        if p['dim'] == p['dim-gm']:
            # MSE, p_in, z, z2 belong together..:
            if (('MSE' in mets) or ('p_in' in mets) or ('z' in mets)
                    or ('z2' in mets)):
                if ('exp' in rule or 'grad' in rule or 'pf' == rule):
                    delta = v['mu'][k:end] - v['lam'][k:end]
                    if 'pf' == rule:
                        S = 1 / np.array([
                            np.diag(v['sig2'][k + i])
                            for i in range(len(delta))
                        ])
                    else:
                        S = v['sig2'][k:end]
                    MSE = delta**2
                    p_in = MSE < S  # number
                    z = delta / S**0.5  # vector
                    z2 = MSE / S  # vector
                elif 'corr' in rule:  # include correlations
                    M, S = ('mu', 'sig2')
                    tmax, dim = v['x'].shape
                    if dim == 1:
                        delta = (v[M][k:end] - v['lam'][k:end]).squeeze()
                        A = 1 / v[S][k:end].squeeze()
                        z2 = A * delta**2
                        z = A**0.5 * delta
                    else:  # dim > 1
                        if tmax > 2:  # not online, precompute!
                            delta = v[M][k:end] - v['lam'][k:end]
                            As = np.linalg.inv(
                                v[S][k:end]) / dim  # will be scalar
                            z = np.array([
                                sqrtm(As[i]).dot(delta[i])
                                for i in range(len(delta))
                            ])
                            z2 = np.array([
                                As[i].dot(delta[i]).dot(delta[i])
                                for i in range(len(delta))
                            ])

                        else:  # online
                            delta = (v[M][k:end] - v['lam'][k:end]).squeeze()
                            A = np.linalg.inv(
                                v[S][k:end].squeeze()) / dim  # will be scalar
                            z = sqrtm(A).dot(delta)
                            z2 = A.dot(delta).dot(delta)
                    MSE = delta**2
                    p_in = z2 < 1
            # fill in
            ro[mets.index('MSE')] = np.mean(MSE)
            ro[mets.index('p_in')] = np.mean(p_in)
            ro[mets.index('z')] = np.mean(z)
            ro[mets.index('z2')] = np.mean(z2)

        ### model mismatch or no model mismatch
        if 'z2d' in mets:
            # catch instability
            if v['gbar'][k:end] > 1:
                p['gbardt>1'] += 1  # warning does not work for offline
                if p['gbardt>1'] == 0:
                    print('step', t, 'set gbardt = 0.99 gbardt=',
                          v['gbar'][k:end])
                v['gbar'][k:end] = 0.99

            # compute log likelihood for all models with g as reference
            if v['y'][k:end] > 0:  # and v['g'][k:end]>0:
                z2d = np.log(v['gbar'][k:end] / p['g0dt'])
            else:  # only useful while ytref = 0
                z2d = np.log((1 - v['gbar'][k:end]) / (1 - p['g0dt']))
            # fill in
            ro[mets.index('z2d')] = np.mean(z2d)

            # pts estimate prediction with SF (exp, corr)
            if (('z2d_pt' in mets) and ('corr' in rule or 'exp' in rule)):
                if v['y'][k:end] > 0:  # and v['g'][k:end]>0:
                    z2d_pt = np.log(v['gmap'][k:end] / p['g0dt'])
                else:  # only useful while ytref = 0
                    z2d_pt = np.log((1 - v['gmap'][k:end]) / (1 - p['g0dt']))
                ro[mets.index('z2d_pt')] = np.mean(z2d_pt)

            # generative prob
            if 'z2dg' in mets:
                if v['y'][k:end] > 0:  # and v['g'][k:end]>0:
                    z2dg = np.log(v['g'][k:end] / p['g0dt'])
                else:  # only useful while ytref = 0
                    z2dg = np.log((1 - v['g'][k:end]) / (1 - p['g0dt']))
                ro[mets.index('z2dg')] = np.mean(z2dg)

        return (ro)

    def run_world(self, p, t, k):
        """ update world parameters in performance sims """
        v = self
        dt = p['dt']

        # run GM
        if p['w-dynamic'] == 'static':
            if 'hetero-STDP':  # set both weights to zero and const
                v['lam'][k + 1][:] = (1, 1) if t == 0 else v['lam'][k]
            else:
                v['lam'][k + 1] = np.random.choice(
                    [-p['sig2_ou'], p['sig2_ou']],
                    p['dim-gm']) if t == 0 else v['lam'][k]
        elif p['w-dynamic'] == 'OU':
            v['lam'][k + 1] = v['lam'][k] + dt / p['tau_ou'] * (
                p['mu_ou'] - v['lam'][k]) + p['dW'] * eta(p['dim-gm'])

        if p['spikes']:
            # standard case (alpha is strange here, has to be 1)
            if p['hetero-STDP'] == False:
                ix = np.where(
                    (v['x_ref'] <= 0) * (t > v['t']) *
                    np.random.binomial(1, 1 +
                                       (p['nu*dt'] - 1) * p['alpha']))[0]
                # select timer according to alpha, e.g. (but not sure)
                v['t'][ix] += (1 - p['alpha']) / p['nu*dt'][ix]
            else:
                if t == 1:
                    print(
                        'Note: init dT of spike trains in run world, then clock'
                    )
                    v['t'][0] = 1
                    v['t'][1] = 1 + 0.5 * p['alpha'] / p['nu*dt'][1]

                ix = np.where((v['x_ref'] <= 0) * (t > v['t']))[0]
                #v['t'][ix] += sample_ISI(p['nu*dt'][ix]/dt,p['alpha'])/dt
                # add to t0 always the same if spike
                v['t'][ix] += 1 / p['nu*dt'][ix]  # steps

            # no spiking for bias!
            if p['bias']:
                ix = ix[ix > 0]
                #v['x'][k,0] = 1

            v['x'][k, ix] += 1  # init cond is zero for entire array

            # sample only for spikes
            if 'smooth' in p['rule'] or 'sample' in p['rule']:
                v.w_sample(k, ix, p)

            v['x'][k + 1] = (1 - dt / p['tau']) * v['x'][k]
            v['x_ref'] -= dt  # count ref period
            v['x_ref'][ix] = p['t_ref']  # re-init for spikes

            if p['dim-gm'] > p['dim']:
                if p['hetero-STDP']:
                    print(
                        'Warning: model mismatch does not work with hetero-STDP'
                    )
                # select timer according to alpha, e.g. (but not sure)
                ix2 = np.where(
                    (v['x_ref-gm'] <= 0) * (t > v['t-gm']) *
                    np.random.binomial(1, 1 +
                                       (p['nu*dt-gm'] - 1) * p['alpha']))[0]
                p['ix2'] = ix2
                ix2 = p['ix2']
                v['t-gm'][ix2] += (1 - p['alpha']) / p['nu*dt-gm'][ix2]
                v['x-gm'][k, ix2] += 1  # init cond is zero for entire array
                v['x-gm'][k + 1] = (1 - dt / p['tau']) * v['x-gm'][k]
                v['x_ref-gm'] -= dt  # count ref period
                v['x_ref-gm'][ix2] = p['t_ref']  # re-init for spikes

        else:  # constant input
            v['x'][k + 1] = p['nu*dt'] / dt

        if p['bias']:
            v['x'][k][0] = 1

        # generation of membrane
        if p['dim-gm'] > p['dim']:  # add extra generation mechansim
            v['u'][k] = v['x'][k].dot(
                v['lam'][k, :p['dim']]) + v['x-gm'][k].dot(
                    v['lam'][k, p['dim']:])
        elif p['dim-gm'] < p['dim']:  # cut production mechansim
            v['u'][k] = v['x'][k, :p['dim-gm']].dot(v['lam'][k, :p['dim-gm']])
        else:  # equal dims
            v['u'][k] = v['x'][k].dot(v['lam'][k])
        v['g'][k] = p['g0dt'] * np.exp(p['beta'] * v['u'][k])

        # catch instability
        if v['g'][k] > 1:
            p['gdt>1'] += 1
            print('step', t, 'set gdt = 1. gdt=', v['g'][k])
            v['g'][k] = 1

        if v['y_ref'] <= 0:
            if p['hetero-STDP']:
                v['y'][k] = 0

            else:
                v['y'][k] = np.random.binomial(1, v['g'][k])
        else:
            v['y'][k] = 0

        # spike with t_ref
        v['y_ref'] = p['ty_ref'] if v['y'][k] else v['y_ref'] - dt


def gen_table(mp):
    """ generate a data frame with all sims to run and output prepred """
    if 'protocol' not in mp['w-dynamic']:
        out = []
        A, B, C, D, E, F, K, M, N, O, J = ('lr', 'w-dynamic', 'dim', 'm', 'L',
                                           'beta', 'rule', 'tau_s', 'beta0',
                                           'alpha', 'dim-gm')
        #G,H,I,J,#
        #L,#

        # run Bayesian sim
        for a, b, c, m, d, k, n, o, j in it.product(mp['lrs'], mp['w-dynamic'],
                                                    mp['dims'], mp['tau_ss'],
                                                    np.arange(mp['M']),
                                                    mp['rules'], mp['beta0s'],
                                                    mp['alphas'],
                                                    mp['dims-gm']):
            out.append({
                A: a,
                B: b,
                C: c,
                D: d,
                E: mp['L'],
                K: k,
                M: m,
                N: n,
                O: o,
                F: np.nan,
                J: j
            })
        out = pd.DataFrame(out)
        for m in mp['mets']:  # metrics
            out[m] = np.nan
        # rm learning rates for Bayesian rules
        out.loc[out.rule != 'grad', 'lr'] = np.nan
        # rm tau_s for non sampling rules
        out.loc[out.rule.apply(lambda s: ('sample' not in s) and
                               ('smooth' not in s)), 'tau_s'] = np.nan
        # match dim-gm generative to dim if not specified
        out.loc[np.isnan(out['dim-gm']
                         ), 'dim-gm'] = out.loc[np.isnan(out['dim-gm']), 'dim']

        # remove learning rate for rules that are not grad
        out.loc[out.rule != 'grad', 'lr'] = np.nan
        out = out.drop_duplicates()
        #out = out[out.dim > 1]
        out.reset_index(inplace=True)
        out.drop(['index'], axis=1, inplace=True)
        if mp['hetero-STDP']:
            o = np.nan
            for key in [
                    'nBs', 'mu0_0s', 'mu0_1s', 'sig20_00s', 'sig20_01s',
                    'sig20_11s'
            ]:
                out = add_key(out, key[:-1], [o])
            for key in ['muf_0', 'muf_1', 'sig2f_00', 'sig2f_01', 'sig2f_11']:
                out = add_key(out, key, [o])
            for key, vals in zip(
                ['dS', 'dT', 'dT2', 'nB'],
                [[mp['dS']], [mp['dT']], [mp['dT2']], [mp['nB']]]):
                out = add_key(out, key, vals)

            # add spikes
            if 'hetero-STDP-xSpikes' in mp:
                out = add_key(out, 'hetero-STDP-xSpikes',
                              mp['hetero-STDP-xSpikes'])

        # report failures
        out['w<0'] = 0
        out['gdt>1'] = 0
        out['gbardt>1'] = 0

    elif 'protocol' in mp['w-dynamic']:
        out = []
        o = np.nan
        dT_range = mp['dT_range']
        dT2 = o
        hcorrs = [0, 1] if mp['hetero-STDP'] else [np.nan]  # precondition?
        for d, nB, tau, m, r, beta, bias, n_PC, hcorr in it.product(
                mp['dims'], mp['nBs'], mp['tau_ss'], np.arange(mp['M']),
                mp['rules'], mp['beta0s'], mp['biass'], mp['n_PCs'], hcorrs):
            # dT
            dT = dT_range * (2 * m / mp['M'] - 1)
            out.append({
                'm': m,
                'nB': nB,
                'nS': mp['nS'],
                'dS': 0.05,
                'bias': bias,
                'dT': dT,
                'dT2': dT2,
                'dim': d,
                'alpha': o,
                'lr': o,
                'tau_s': tau,
                'beta': beta,
                'beta0': 1,
                'rule': r,
                'w-dynamic': mp['w-dynamic'][0],
                'n_PC': n_PC,
                'hetero-correlations': hcorr
            })
        out = pd.DataFrame(out)

        # errors
        out['w<0'] = 0
        out['gdt>1'] = 0
        out['gbardt>1'] = 0

        # add general stuff
        if 'hetero-STDP-xSpikes' in mp:
            out = add_key(out, 'hetero-STDP-xSpikes',
                          mp['hetero-STDP-xSpikes'])

        # add mean: suffix and time for readout
        for s, tt in it.product(['b', '1', '2'], ['0', 'i', 'f']):
            key = 'mu{0}_{1}'.format(tt, s)
            if tt == '0':  # add init cond
                out = add_key(out, key, mp[key + 's'])
            else:  # add dummy for readout
                out[key] = np.nan

        # add covariance
        for s, tt in it.product(['b', 'w', 'bw', 'ww'], ['0', 'i', 'f']):
            key = 'Sig{0}_{1}'.format(tt, s)
            # add init cond
            if tt == '0':
                out = add_key(out, key, mp[key + 's'])
                # rm some
                if (s == 'bw' or s == 'ww'):  # only diagonal
                    out.loc[out.rule.apply(lambda rr: 'exp' in rr
                                           ), key] = np.nan
                if (s == 'bw'):  # no cross talk
                    out.loc[out.rule.apply(lambda rr: 'corrx' in rr
                                           ), key] = np.nan
            else:  # add dummy for readout
                out[key] = np.nan

        # add 3rd spike
        if mp['triplet']:
            dTs = np.hstack(
                [np.nan, dT_range * (2 * np.arange(mp['M']) / mp['M'] - 1)])
            for spike3 in ('dT_pre2', 'dT_post2'):
                out = add_key(out, spike3, dTs)

            # remove entries with 4 spikes => at least one nan required
            out = out.loc[(np.isnan(out['dT_pre2'])) |
                          (np.isnan(out['dT_post2']))]

            # remove lower triangles
            out = out.loc[(out['dT_pre2'] > out['dT'])
                          | np.isnan(out['dT_pre2'])]
            out = out.loc[(out['dT_post2'] > out['dT'])
                          | np.isnan(out['dT_post2'])]
            # remove STDP trials
            out = out.loc[(np.isnan(out['dT_pre2'])
                           & np.isnan(out['dT_post2'])) == False]

        # remove additional useless combinations
        if (1 in out.dim.unique()) and (1 in out.bias.unique()):
            #out.loc[((out.dim==1) & (out.bias==1))] = np.nan
            out.loc[((out.dim == 1) & (out.bias == 1)), 'bias'] = 0
            # remove all other rules
            out.loc[((out.dim == 1) & (out.bias == 0) &
                     (out.rule != 'exp'))] = np.nan

            #print('bias and dim = 1 accepted currently')
        if (3 in out.dim.unique()) and (0 in out.bias.unique()):
            out.loc[(out.dim == 3 & out.bias == 0)] = np.nan  # too many dims
        out = out.drop_duplicates()
        out = out.dropna(thresh=1)
        out.reset_index(inplace=True)
        out.drop(['index'], axis=1, inplace=True)
    return (out)


def rm_sims(out, thres=0.01):
    """ remove flawed sims and print how error statistics """
    #max_err = thres*total_steps
    #total_steps = 10*100/0.0005 # fig 2,3 not pf
    #total_steps = 200*5/0.0005 # fig 4

    for r, b, d in it.product(out.rule.unique(), out.beta0.unique(),
                              out.dim.unique()):
        # select
        out_r = out[(out.rule == r) & (out.beta0 == b) & (out.dim == d)]
        if len(out_r) > 0:
            print(r, 'beta:', b, 'dim', d)
            if 'pf' in r:
                total_steps = 10 * 100 / 0.001  # fig 2,3 pf
            else:
                total_steps = 200 * 5 / 0.0005  # fig 2,3,4 not pf
            max_err = thres * total_steps

            for err in ('gbardt>1', 'gdt>1', 'w<0'):
                print(err, 'frac:',
                      out_r[err].sum() / (total_steps * len(out_r)), 'sum:',
                      out_r[err].sum())
                #print('sims to be dropped')

            out2_r = out_r[(out_r['gbardt>1'] < max_err)
                           & (out_r['gdt>1'] < max_err) &
                           (out_r['w<0'] < max_err)]
            print('maintain:', len(out2_r), '/', len(out_r))
            print(' ')

    max_err = thres * total_steps
    out2 = out[(out['gbardt>1'] < max_err) & (out['gdt>1'] < max_err) &
               (out['w<0'] < max_err)]
    return (out2)


def tab2v(tab, sim_id, v, p, k=0, lab='0', rev=False):
    """ fct to load and unload v and tab:
        tab2v(0,v,p) # init v
        tab2v(0,v,p,k=k,lab='i',rev=True) # before STDP
        tab2v(0,v,p,k=k,lab='f',rev=True) # after STDP
    """

    if rev == False:  # from tab to v

        # dim == 1
        if p['dim'] == 1 and p['bias'] == True:
            v['mu'][k] = tab.loc[sim_id]['mu0_b']
            v['sig2'][k] = tab.loc[sim_id]['Sig0_b']
            print('dim=1,bias=1 shouldnt exist... buuugg!')
        if p['dim'] == 1 and p['bias'] == False:
            v['mu'][k] = tab.loc[sim_id]['mu0_1']
            v['sig2'][k] = tab.loc[sim_id]['Sig0_w']

        # dim == 2
        if p['dim'] == 2:
            m1, m2, s1, s2, s3 = (('b', '1', 'b', 'w', 'bw') if p['bias'] else
                                  ('1', '2', 'w', 'w', 'ww'))

            v['mu'][k] = (tab.loc[sim_id]['mu0_' + m1],
                          tab.loc[sim_id]['mu0_' + m2])

            if 'exp' in p['rule']:
                v['sig2'][k] = (tab.loc[sim_id]['Sig0_' + s1],
                                tab.loc[sim_id]['Sig0_' + s2])

            elif 'corr' in p['rule']:
                v['sig2'][k] = np.array([[
                    tab.loc[sim_id]['Sig0_' + s1],
                    tab.loc[sim_id]['Sig0_' + s3]
                ],
                                         [
                                             tab.loc[sim_id]['Sig0_' + s3],
                                             tab.loc[sim_id]['Sig0_' + s2]
                                         ]])

        # dim == 3
        if p['dim'] == 3:
            v['mu'][k] = (tab.loc[sim_id, 'mu0_b'], tab.loc[sim_id, 'mu0_1'],
                          tab.loc[sim_id, 'mu0_2'])
            if 'exp' in p['rule']:
                v['sig2'][k] = (tab.loc[sim_id, 'Sig0_b'],
                                tab.loc[sim_id, 'Sig0_w'],
                                tab.loc[sim_id, 'Sig0_w'])
            elif 'corr' in p['rule']:
                row1 = [
                    tab.loc[sim_id, 'Sig0_b'], tab.loc[sim_id, 'Sig0_bw'],
                    tab.loc[sim_id, 'Sig0_bw']
                ]
                row2 = [
                    tab.loc[sim_id, 'Sig0_bw'], tab.loc[sim_id, 'Sig0_w'],
                    tab.loc[sim_id, 'Sig0_ww']
                ]
                row3 = [
                    tab.loc[sim_id, 'Sig0_bw'], tab.loc[sim_id, 'Sig0_ww'],
                    tab.loc[sim_id, 'Sig0_w']
                ]
                v['sig2'][k] = np.array([row1, row2, row3])

    else:  # extract
        mu = v['mu'][k]
        Sig = v['sig2'][k]

        if lab == '0':
            print('lab needs to be i (inbetween) or f (final), but lab=0')
            []**2
        # dim == 1
        if p['dim'] == 1 and p['bias'] == True:
            tab.loc[sim_id, 'mu{0}_b'.format(lab)] = mu
            tab.loc[sim_id, 'Sig{0}_b'.format(lab)] = Sig
            print('dim=1,bias=1 shouldnt exist... buuugg!')
        if p['dim'] == 1 and p['bias'] == False:
            tab.loc[sim_id, 'mu{0}_1'.format(lab)] = mu
            tab.loc[sim_id, 'Sig{0}_w'.format(lab)] = Sig

        # dim == 2
        if p['dim'] == 2:
            i1, i2 = int(p['bias']), int(1 + p['bias'])  #1st,2nd weight idx
            tab.loc[sim_id, 'mu{0}_b'.format(lab)] = mu[0]
            tab.loc[sim_id, 'mu{0}_1'.format(lab)] = mu[i1]  # lower if b=0
            if p['bias'] == False:  # if bias, 2nd weight DNE
                tab.loc[sim_id, 'mu{0}_2'.format(lab)] = mu[i2]

            if 'exp' in p['rule']:
                tab.loc[sim_id, 'Sig{0}_b'.format(lab)] = Sig[0]
                tab.loc[sim_id, 'Sig{0}_w'.format(lab)] = Sig[i1]
            else:
                tab.loc[sim_id, 'Sig{0}_b'.format(lab)] = Sig[0, 0]
                tab.loc[sim_id, 'Sig{0}_w'.format(lab)] = Sig[i1, i1]
                tab.loc[sim_id, 'Sig{0}_bw'.format(lab)] = Sig[1, 0]
                if p['bias'] == False:  # if bias, 2nd weight DNE
                    tab.loc[sim_id, 'Sig{0}_ww'.format(lab)] = Sig[i1, i2]
            # ignore 2nd weight

        # dim == 3
        if p['dim'] == 3:
            tab.loc[sim_id, 'mu{0}_b'.format(lab)] = mu[0]
            tab.loc[sim_id, 'mu{0}_1'.format(lab)] = mu[1]
            tab.loc[sim_id, 'mu{0}_2'.format(lab)] = mu[2]

            if 'exp' in p['rule']:
                tab.loc[sim_id, 'Sig{0}_b'.format(lab)] = Sig[0]
                tab.loc[sim_id, 'Sig{0}_w'.format(lab)] = Sig[1]
            else:
                tab.loc[sim_id, 'Sig{0}_b'.format(lab)] = Sig[0, 0]
                tab.loc[sim_id, 'Sig{0}_w'.format(lab)] = Sig[1, 1]
                tab.loc[sim_id, 'Sig{0}_bw'.format(lab)] = Sig[1, 0]
                tab.loc[sim_id, 'Sig{0}_ww'.format(lab)] = Sig[1, 1]
            # ignore 2nd weight


#### Shifting the Ellipse
def plt_ellipse(mean, cov, ax=None, n=3.0, c='none', **kwargs):
    n_std = n
    ax = plt.gca() if ax is None else ax
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = matplotlib.patches.Ellipse((0, 0),
                                         width=ell_radius_x * 2,
                                         height=ell_radius_y * 2,
                                         facecolor=c,
                                         **kwargs)
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = mean[0]

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = mean[1]

    transf = matplotlib.transforms.Affine2D().rotate_deg(45).scale(
        scale_x, scale_y).translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


def plt_time_series(fig=None, v=None, p=None, t=None, text_size=13):
    """ if fig provided, plot time series shown in manuscript
        else: look for v,p,t and do diagnostic plot of variables

        fig (str):
            fig1d => TS of w and p(w|D)
            figS4 => STDP TS
            figS5 => Hetero TS                
    """

    if fig is not None:
        if fig == 'fig1d':
            lw = 2
            fig1d = load_obj('./pkl_data/fig1d/fig1d')
            plt.plot(fig1d['tspan'],
                     fig1d['yplt_gt'],
                     'k',
                     linewidth=lw,
                     label='Ground truth')
            plt.fill_between(fig1d['tspan'],
                             fig1d['yplt'] - fig1d['err'],
                             fig1d['yplt'] + fig1d['err'],
                             alpha=0.3,
                             color='r')
            plt.plot(fig1d['tspan'],
                     fig1d['yplt'],
                     'r',
                     linewidth=lw,
                     label='Synaptic Filter')

            plt_legend()
            plt.xlabel(r'Time $t$ [s]')
            plt.ylabel(r'Weights $w$')
            [plt.locator_params(axis=axis, nbins=2) for axis in ['x', 'y']]
            #plt.gca().set_xticks([0,T])
            plt.title(r'Weight 1 ($d=2$, $\beta_0 = 1$, $\tau_{\rm ou}=100$s)')
            plt.savefig('./figures/fig1d.pdf', dpi=300, bbox_inches='tight')

        elif fig == 'figS4':

            for suf in ('prepost', 'postpre'):

                f, axs = plt.subplots(3, 1, sharex=True, figsize=[6, 8])
                lw = 2
                #dt = p['dt']
                #i0, i1 = int(0.3 / dt), int(0.4 / dt)
                figS4 = load_obj('./pkl_data/figS4/figS4_' + suf)
                tspan = figS4['tspan']
                bias = figS4['bias']
                mu1 = figS4['mu1']
                cov0 = figS4['cov0']
                cov1 = figS4['cov1']
                cov12 = figS4['cov12']
                x1 = figS4['x1']
                y0 = figS4['y0']
                dT = figS4['dT']

                plt.sca(axs[0])
                if dT < 0:
                    tit = r'Negative STDP lobe, $t_{\rm post} < t_{\rm pre}$'
                else:
                    tit = r'Positive STDP lobe, $t_{\rm pre} < t_{\rm post}$'
                plt.title(tit)
                plt.plot(tspan,
                         x1,
                         color='black',
                         lw=lw,
                         label=r'Presyn. activation')
                plt.plot(tspan, y0, color='red', lw=lw, label='Postsyn. spike')
                plt.ylim([-0.05, 1.3])
                [plt.locator_params(axis=axis, nbins=3) for axis in ['x', 'y']]
                plt_legend(ncol=2, prop={'size': text_size})
                plt.ylabel('Spikes')
                #plt.tight_layout()

                # mean:
                plt.sca(axs[1])
                plt.plot(tspan,
                         bias,
                         color='red',
                         lw=lw,
                         label=r'Mean bias $\mu_{t,0}$')
                plt.plot(tspan,
                         mu1,
                         color='black',
                         lw=lw,
                         label=r'Mean weight $\mu_{t,1}$')
                plt.ylim([-0.05, None])
                [plt.locator_params(axis=axis, nbins=3) for axis in ['x', 'y']]
                plt_legend(ncol=2, prop={'size': text_size})
                plt.ylabel(r'Weight means $\mu_t$')
                plt.xlabel(r'Time $t$ [s]')

                plt.sca(axs[2])
                plt.plot(tspan,
                         cov0,
                         color='red',
                         lw=lw,
                         label=r'Bias var. $\Sigma_{t,00}$')
                plt.plot(tspan,
                         cov1,
                         color='black',
                         lw=lw,
                         label=r'Weight var. $\Sigma_{t,11}$')
                plt.plot(tspan,
                         cov12,
                         color='gray',
                         lw=lw,
                         label=r'Cov. $\Sigma_{t,01}$')
                plt.ylim([None, None])
                [plt.locator_params(axis=axis, nbins=3) for axis in ['x', 'y']]
                plt_legend(ncol=1, prop={'size': text_size}, loc=10)
                plt.ylabel(r'Covariance $\Sigma_t$')
                plt.xlabel(r'Time $t$ [s]')
                #plt.gca().set_xticks([0,0])

                plt.tight_layout()
                name = 'figS4_{0}{1}.pdf'.format('STDPtrace', round(dT, 5))
                plt.savefig('./figures/' + name, dpi=300, bbox_inches='tight')
                plt.show(), plt.close()

        elif fig == 'figS5':

            for suf in ('prepost', 'postpre'):

                f, axs = plt.subplots(3, 1, sharex=True, figsize=[6, 8])
                lw = 2

                figS5 = load_obj('./pkl_data/figS5/figS5_' + suf)
                tspan = figS5['tspan']
                bias = figS5['bias']
                mu1 = figS5['mu1']
                mu2 = figS5['mu2']
                cov0 = figS5['cov0']
                cov1 = figS5['cov1']
                cov12 = figS5['cov12']
                x1 = figS5['x1']
                x2 = figS5['x2']
                y0 = figS5['y0']
                dT = figS5['dT']

                plt.sca(axs[0])
                if dT < 0:
                    tit = r'Pos. heterosyn. plasticity, $t_{\rm post} < t_{\rm pre}$'
                else:
                    tit = r'Neg. heterosyn. plasticity, $t_{\rm pre} < t_{\rm post}$'
                plt.title(tit)
                plt.plot(tspan,
                         x1,
                         color='black',
                         lw=lw,
                         label=r'Presyn. activation $x^\epsilon_{t,1}$')
                plt.plot(tspan,
                         x2,
                         color='gray',
                         lw=1,
                         label=r'Presyn. activation $x^\epsilon_{t,2}$')
                plt.plot(tspan, y0, color='red', lw=lw, label='Postsyn. spike')
                plt.ylim([-0.05, None])
                [plt.locator_params(axis=axis, nbins=3) for axis in ['x', 'y']]
                plt_legend(ncol=1, prop={'size': text_size}, loc=9)
                plt.ylabel('Spikes')
                #plt.tight_layout()

                # mean:
                plt.sca(axs[1])

                plt.plot(tspan,
                         bias,
                         color='red',
                         lw=lw,
                         label=r'Bias $\mu_{t,0}$')
                plt.plot(tspan,
                         mu1,
                         color='black',
                         lw=lw,
                         label=r'Weight $\mu_{t,1}$')
                plt.plot(tspan,
                         mu2,
                         color='gray',
                         lw=lw,
                         label=r'Weight $\mu_{t,2}$')
                plt.ylim([-0.5, None])
                [plt.locator_params(axis=axis, nbins=3) for axis in ['x', 'y']]
                plt_legend(ncol=2, prop={'size': text_size}, loc=3)
                plt.ylabel(r'Weight means $\mu_t$')

                # var
                plt.sca(axs[2])

                plt.plot(tspan,
                         cov0,
                         color='red',
                         lw=lw,
                         label=r'Bias var. $\Sigma_{t,00}$')
                plt.plot(tspan,
                         cov1,
                         color='black',
                         lw=lw,
                         label=r'Weight var. $\Sigma_{t,11}$')
                plt.plot(tspan,
                         cov12,
                         color='gray',
                         lw=lw,
                         label=r'Weight cov. $\Sigma_{t,12}$')
                plt.ylim([None, None])
                [plt.locator_params(axis=axis, nbins=3) for axis in ['x', 'y']]
                plt_legend(ncol=1, prop={'size': text_size}, loc=10)
                plt.ylabel(r'Covariance $\Sigma_t$')
                plt.xlabel(r'Time $t$ [s]')
                #plt.gca().set_xticks([0,0])

                plt.tight_layout()
                name = 'figS5_{0}{1}.pdf'.format('HETEROtrace', round(dT, 5))
                plt.savefig('./figures/' + name, dpi=300, bbox_inches='tight')
                plt.show(), plt.close()

    else:  # diagnostic
        # protocol:
        #if sim_id == 0:
        f, axs = plt.subplots(3, 1, sharex=True, figsize=[6, 8])
        plt.sca(axs[0])
        v.plt('x', 0), plt.title(
            str(p['m']) + ' ' + p['rule'] + ', dT=' + str(round(p['dT'], 4)) +
            ', dT_pre2=' + str(round(p['dT_pre2'], 4)) + ', dT_post2=' +
            str(round(p['dT_post2'], 4)))

        if p['dim'] > 1:
            v.plt('x', 1)
            if p['dim'] > 2:
                v.plt('x', 2)
        v.plt('y', 0), plt.xlabel('')  #, plt.show(), plt.close()
        # read out
        xreadout = int(p['T_wait'] / 2 / p['dt'] - 1) * p['dt']
        plt.plot([xreadout, xreadout], [0, 1], ':k', lw=3, label='readout')

        plt.xlim([0, (t - 1) * p['dt']])

        # mean:
        plt.sca(axs[1])
        v.plt('mu', 0)  #, plt.title('Mean, '+p['rule'])
        if p['dim'] > 1:
            v.plt('mu', 1)
            if p['dim'] > 2:
                v.plt('mu', 2)
        plt.xlabel('')  #, plt.show(), plt.close()
        plt.xlim([0, (t - 1) * p['dt']])

        # cov:
        plt.sca(axs[2])
        v.plt('sig2', 0)  #, plt.ylabel('Cov, '+p['rule'])
        if p['dim'] > 1:
            v.plt('sig2', 1)
            if p['dim'] > 2:
                v.plt('sig2', 2)
                plt_legend()
        plt.xlim([0, (t - 1) * p['dt']])
        plt.show(), plt.close()


def plt_manuscript_figures(fig, mp, text_size=13):
    """ 
        load data from ./pkl_data/ and plot in ./figures/
    """

    # define names and plotting variables
    m2l = {
        'MSE': 'MSE',
        'p_in': 'Ground truth within $\sigma$: $p_{in}$',
        'z': 'First moment $z^{(1)}$',
        'z2': 'Second moment $z^{(2)}$',
        'z2d': 'diagonal second moment $z^2_d$'
    }
    x2l = {
        'beta0': r'Determinism $\beta_0$',
        'dim': 'Dimension $d$',
        'lr': r'Learning rate $\eta$'
    }
    me2l = {'beta0': r'$\beta_0$', 'dim': r'$d$'}
    # rule to style and color
    r2s = {
        'exp': '-',
        'corr': '-',
        'exp_smooth': ':',
        'corr_smooth': ':',
        'pf_corr': '-'
    }
    r2c = {
        'exp': 'k',
        'corr': 'r',
        'exp_smooth': 'k',
        'corr_smooth': 'r',
        'pf_corr': 'gray',
        'grad': 'gray'
    }
    r2l = {
        'exp': 'Diag. SF',  # 'DSF',
        'corr': 'Syn. Fil.',  #'SF',
        'exp_smooth': 'Diag. Samp. SF',  # 'DSF',
        'corr_smooth': 'Samp. SF',  # 'DSF',
        'exp_sample': r'$D(\Sigma), u^s$',
        'corr_sample': r'$\Sigma, u^s$',
        'grad': 'Gradient',
        'pf_corr': 'PF'
    }

    # time series
    if fig in ('figS4', 'figS5', 'fig1d'):
        plt_time_series(fig, text_size=text_size)

    elif 'fig3' in fig:
        """ STDP curves """  # jjjj

        # load STDP
        out = load_obj('./pkl_data/fig3/fig3')

        r2s = {
            'exp': '-',
            'corr': '-',
            'exp_smooth': ':',
            'corr_smooth': ':',
            'grad': '-'
        }
        xlab = 'post-pre delay $t_{post}-t_{pre}$ [s]'
        ylabs = [r'Weight change d$\mu$', r'Variance change d$\sigma^2$']
        tits = ['STDP Mean', 'STDP Variance']
        # three rules: corr, exp, bias=False
        rules, biass, cs = ['exp', 'exp',
                            'corr'], [False, True, True], ['gray', 'k', 'red']
        labs = ['No bias', 'Bias, no correlation', 'Bias, correlation']

        for key in [0, 1]:  # mu, var
            for i in range(3):
                #if 1:
                r, bias, c, lab = rules[i], biass[i], cs[i], labs[i]
                ylab, tit = ylabs[key], tits[key]
                outc = out[(out.rule == r) & (out.bias == bias)]
                if key == 0:
                    yplt = outc.muf_1 - outc.mui_1
                elif key == 1:
                    yplt = outc.Sigf_w - outc.Sigi_w
                plt.title(tit)
                plt.plot(outc.dT,
                         yplt,
                         lw=2,
                         label=lab,
                         linestyle=r2s[r],
                         color=c)
                plt.plot(outc.dT, np.zeros(len(outc.dT)), 'k:', alpha=0.5)
                [plt.locator_params(axis=axis, nbins=3) for axis in ['x', 'y']]
                plt.xlabel(xlab)
                plt.ylabel(ylab)
                #ylims.append(plt.gca().get_ylim())
            plt_legend(ncol=1, prop={'size': text_size})
            plt.tight_layout()
            plt.savefig('./figures/fig3_{0}.pdf'.format(tit),
                        dpi=300,
                        bbox_inches='tight')
            plt.show(), plt.close()

    if 'fig4' in fig:
        """ hetero / homo synaptic plasticity curves """
        # load for plotting
        tab = load_obj('./pkl_data/fig4/fig4')

        ## relative weight change
        st2lab = {
            'hetero': 'Heterosynaptic',
            'mixed': 'mixed',
            'homo': 'Homosynaptic'
        }
        print('just for now switch homo and hetero and color!!')
        pc2lab = {0: 'no PC', 1: 'PC'}
        pc2ls = {0: '--', 1: '-'}
        #r2t = {'corr':'SF', 'exp':'dSF','corrx':'bSF'}
        r2t = {
            'corr': 'Synaptic Filter (correlations)',
            'corrx': 'Block Synaptic Filter',
            'exp': 'Diagonal Synaptic Filter (no corr.)'
        }
        # num of preconditioning spikes
        n_PC = mp['n_PCs'][1] if 'n_PC' not in mp else mp['n_PC']
        st2lab = {2: 'Heterosyn.', 1: 'Homosyn.'}
        for r in tab.rule.unique():
            ylab = r'Weight change $\Delta \mu$'
            for spikeType, c in zip([1, 2], ('k', 'r')):
                for pc in [0, 1]:
                    out2 = tab[(tab['hetero-STDP-xSpikes'] == 'homo')
                               & (tab['hetero-correlations'] == pc) &
                               (tab['rule'] == r) & (tab['n_PC'] == n_PC)]

                    out2['dmu'] = out2['muf_' + str(spikeType)].sub(
                        out2['mui_' + str(spikeType)], axis=0)  # - 1
                    yplt = out2['dmu'].values  #out2.groupby(['alpha']).mean()
                    xplt = out2.dT.unique()
                    baseline = 0  #if pc == 0 else yplt[0] # LTD component
                    lab = st2lab[
                        spikeType] if pc == 0 else st2lab[spikeType] + ' + PC'
                    plt.plot(xplt,
                             yplt - baseline,
                             c=c,
                             ls=pc2ls[pc],
                             lw=2,
                             label=lab)
                #plt.ylim([-0.65, 0.65])
                #plt.plot(xplt,xplt*0,lw=2,c='k',ls='--',alpha=0.6)

            #plt.xlabel(r'Spike train phase shift, $\Delta T$ [ms]')
            plt.xlabel(
                r'Spike time difference, $t_{\rm{post}}^{} - t_{\rm{pre}}^{}$ [s]'
            )
            plt.ylabel(ylab)
            [plt.locator_params(axis=axis, nbins=2) for axis in ['x', 'y']]

            #plt.gca().set_yticks([-0.3,0,.3])
            #plt.ylim([-1,1.0])
            plt.title(r2t[r])
            ncol = 2  # if pane==1 else 3
            plt.legend(ncol=ncol, prop={'size': 1 * text_size}, loc=3)
            plt.savefig('./figures/' + 'fig4cd_dmu_{0}.pdf'.format(r),
                        dpi=300,
                        bbox_inches='tight')
            plt.show(), plt.close()

        #### LTP LTD corr plot

        # select certain m
        # select homo-LTP, hetero-LTD
        r = 'corr'
        m_max = int(len(tab.m.unique()) / 2 + 1)
        out = tab
        out['dmu1'] = out['muf_1'].sub(out['mui_1'], axis=0)
        out['dmu2'] = out['muf_2'].sub(out['mui_2'], axis=0)
        out2 = out.loc[((out['hetero-STDP-xSpikes'] == 'homo') &
                        (out['hetero-correlations'] == 1) &
                        (out['rule'] == r)), ['dT', 'dmu1', 'dmu2', 'n_PC']]
        out2 = out2[out2.n_PC == 2]
        X, Y = out2.dmu1.values, out2.dmu2.values
        plt.scatter(X, Y, c='k', label='Synaptic Filter')
        [plt.locator_params(axis=axis, nbins=2) for axis in ['x', 'y']]
        xlim = np.array(plt.gca().get_xlim())
        ylim = np.array(plt.gca().get_ylim())

        # fit
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.plot([0, 0], ylim, ':k', alpha=0.5)
        plt.plot(xlim, [0, 0], ':k', alpha=0.5)
        fit_obj = Fit(x=X, y=Y)
        fit_obj.fit(lambda x, th: th[0] + x * th[1], [0.5, -1])
        fit_obj.plt(lw=2,
                    c='r',
                    label='Slope: {0}'.format(round(fit_obj.th[1], 2)))
        plt.xlim(xlim / 0.9)
        plt.ylim(ylim / 0.9)

        # exp
        if 0:
            r = 'exp'
            out2 = out.loc[(
                (out['hetero-STDP-xSpikes'] == 'homo') &
                (out['hetero-correlations'] == 1) &
                (out['rule'] == r)), ['dT', 'dmu1', 'dmu2', 'n_PC']]
            out2 = out2[out2.n_PC == 2]
            X, Y = out2.dmu1.values, out2.dmu2.values
            plt.scatter(X, Y, c='gray', label='Diag. Synaptic Filter')
            plt_legend(loc=3)
        else:
            plt_legend()

        plt.xlabel(r'Homosynaptic plasticity $\Delta \mu_{\rm homo}$')
        plt.ylabel(r'Heterosynaptic plasticity $\Delta \mu_{\rm hetero}$')
        plt.title(r'Simulated data for various $\Delta T$')
        plt.savefig('./figures/fig4e_LTP_vs_LTD{0}.pdf'.format(r),
                    dpi=300,
                    bbox_inches='tight')
        plt.show(), plt.close()

        if 1:  # Data plot from Nature
            # load data set from Pare et Royer [Nature 2003]
            X, Y = np.array([[-0.7034482758620693, 0.7761194029850744],
                             [-0.40000000000000013, 0.8731343283582089],
                             [-0.44827586206896575, 0.7686567164179103],
                             [-0.37241379310344835, 0.7164179104477608],
                             [-0.5862068965517242, 0.18656716417910424],
                             [-0.15172413793103479, 0.23134328358208922],
                             [-0.1586206896551725, 0.05223880597014885],
                             [0.28965517241379324, -0.0820895522388061],
                             [0.15172413793103434, -0.36567164179104505],
                             [0.6068965517241383, -0.43283582089552297],
                             [0.4896551724137934, -0.5671641791044784],
                             [0.9103448275862069, -0.5597014925373136],
                             [0.9103448275862069, -1.4925373134328361],
                             [1.772413793103448, -0.9104477611940305]]).T
            #plt.scatter(X,Y,color='k') #,label='Pare et Roger')
            #plt.title('Data from Royer and Paré [2003]')
            plt.title('Plasticity in BLA to ITC projections')
            plt.scatter(X, Y, c='k', label='Experiments')
            fit_obj = Fit(x=X, y=Y)
            fit_obj.fit(lambda x, th: th[0] + x * th[1], [0.5, -1])
            fit_obj.plt(lw=2,
                        c='r',
                        label='Slope: {0}'.format(round(fit_obj.th[1], 2)))
            #[plt.locator_params(axis=axis, nbins=2) for axis in ['x','y']]
            plt.gca().set_xticks([0, 1]), plt.gca().set_yticks([-1, 0, 1])
            xlim, ylim = plt.gca().get_xlim(), plt.gca().get_ylim()
            plt.plot([0, 0], ylim, ':k', alpha=0.5)
            plt.plot(xlim, [0, 0], ':k', alpha=0.5)
            plt.xlim(xlim)
            plt.ylim(ylim)
            plt_legend()
            plt.xlabel(r'Homosynaptic plasticity $\Delta w_{\rm homo}$')
            plt.ylabel(r'Heterosynaptic plasticity $\Delta w_{\rm hetero}$')
            plt.savefig('./figures/fig4f_data_royer_pare.pdf',
                        dpi=300,
                        bbox_inches='tight')

    # MSE
    if (fig in ('fig2a', 'fig2b', 'fig2ab') or ('figS1' in fig)):

        folders = ['fig2_dim', 'fig2_beta']
        out = folders2df(folders)
        y = 'MSE'
        err_bars = 'std'
        rules = ['exp', 'corr'] if (
            'fig2' in fig) else ['exp', 'corr', 'exp_smooth', 'corr_smooth']
        out = out.sort_index()

        # meta to marker
        m2m = {0: 'o', 1: 'x', 2: 'v'}
        tau_s = 0.1
        ntau_s = 0.1 if tau_s == 1 else 1

        # take dim or beta as x-axis
        for dim_first in [1, 0]:
            meta, x = (('dim', 'beta0'), ('beta0', 'dim'))[dim_first]
            metas = [5] if meta == 'dim' else [1]  # 2nd sim

            # Plot
            plt.ylabel(m2l[y])
            for i, r in it.product(np.arange(len(metas)), rules):
                m = metas[i]
                xyplt = out.loc[(out.rule == r) & (out[meta] == m) &
                                (out.tau_s != ntau_s)]
                if len(out.m.unique()) > 1:  #multi run
                    M = xyplt.groupby(x).mean()
                    S = xyplt.groupby(x).sem()
                    plt.errorbar(M.index,
                                 M[y],
                                 S[y],
                                 c=r2c[r],
                                 ls=r2s[r],
                                 lw=2,
                                 marker='o',
                                 label=r2l[r])
                else:  # single run with corr
                    xyplt = xyplt.sort_values(x)
                    xyplt = xyplt.groupby(x).mean().reset_index(inplace=False)
                    if err_bars:
                        plt.errorbar(xyplt[x],
                                     xyplt[y],
                                     xyplt[y + '_' + err_bars],
                                     c=r2c[r],
                                     marker=m2m[i],
                                     lw=2,
                                     linestyle=r2s[r],
                                     label=r2l[r])
                    else:
                        plt.plot(xyplt[x],
                                 xyplt[y],
                                 c=r2c[r],
                                 marker=m2m[i],
                                 lw=2,
                                 linestyle=r2s[r],
                                 label=r2l[r])

            [plt.locator_params(axis=axis, nbins=2) for axis in ['x', 'y']]
            if x == 'dim':
                plt.gca().set_xticks([1, 15])  #,20])
            elif x == 'beta0':
                plt.gca().set_xticks([0, 1, 2])
            tit = x2l[meta] + '=' + str(m)
            plt.title(tit)
            plt.xlabel(x2l[x])
            plt.ylim([0, None])
            plt_legend(ncol=2, prop={'size': text_size})
            prefix = 'fig2ab' if 'fig2' in fig else 'figS1'
            plt.savefig('./figures/{2}_{1}_x{0}.pdf'.format(x, y, prefix),
                        dpi=300,
                        bbox_inches='tight')
            plt.show(), plt.close()

    elif fig == 'fig2c':  # eta, fig3C jjjj

        # load
        folders = ['fig2_dim', 'fig2_beta', 'fig2_eta']
        out = folders2df(folders)  # check with M

        tit = 'grad_rule'
        errbars = 'std_bar'

        c2ls = {1: '-', 2: '--', 0: ':'}
        c2m = {1: 'o', 2: 'v', 0: '+'}
        tau_s = 1
        lw = 2
        y = 'MSE'
        rules = ['exp', 'corr', 'exp_smooth', 'corr_smooth', 'grad']
        beta0 = 1
        dim = 15

        ## all in one
        rules = ['corr', 'grad']

        plt.figure(figsize=(10, 4))
        for beta0, dim, cond in zip([0.33, 1, 1], [5, 5, 15], [0, 1, 2]):
            for r in rules:
                out2 = out.loc[(out.dim == dim) & (out.beta0 == beta0) & (
                    out.rule == r), ['lr', y, y + '_std', 'dim', 'beta0', 'm']]
                ## with M
                if len(out2.m.unique()) > 1:
                    M = out2.groupby(
                        ['lr']).mean() if r == 'grad' else out2.mean()
                    S = out2.groupby(['lr'
                                      ]).sem() if r == 'grad' else out2.sem()
                    if r == 'grad':
                        xplt = M.index
                        yplt = M[y]
                        yerr = S[y]
                        #ls = '--'
                        lab = r'$\beta_0=$' + str(beta0) + r', $d=$' + str(
                            dim) + ', ' + r2l[r]
                        #lab = None
                    else:
                        xplt = expspace(out.lr.min(), out.lr.max(),
                                        len(out.lr.unique()) - 1)
                        yplt = M[y] * np.ones(len(xplt))
                        yerr = S[y] * np.ones(len(xplt))
                        lab = r'$\beta_0=$' + str(beta0) + r', $d=$' + str(
                            dim) + ', ' + r2l[r]
                        #ls = '-'
                    plt.errorbar(xplt,
                                 yplt,
                                 yerr,
                                 c=r2c[r],
                                 marker=c2m[cond],
                                 ls=c2ls[cond],
                                 lw=lw,
                                 label=lab)

                ## with Corr
                else:
                    if r == 'grad':
                        xplt = out2.lr
                        yplt = out2.MSE
                        yerr = out2[y + '_std']
                        #ls = '--'
                        lab = None
                        lab = r'$\beta_0=$' + str(beta0) + r', $d=$' + str(
                            dim) + ', Gradient'
                    else:
                        xplt = expspace(out.lr.min(), out.lr.max(),
                                        len(out.lr.unique()) - 1)
                        yplt = out2.MSE.values[0] * np.ones(len(xplt))
                        yerr = out2[y + '_std'].values[0] * np.ones(len(xplt))
                        lab = r'$\beta_0=$' + str(beta0) + r', $d=$' + str(
                            dim) + ', SF'
                        #ls = '-'
                    plt.errorbar(xplt,
                                 yplt,
                                 yerr,
                                 c=r2c[r],
                                 marker=c2m[cond],
                                 ls=c2ls[cond],
                                 lw=lw,
                                 label=lab)

        plt.ylim([0.0, 1.2])
        [plt.locator_params(axis=axis, nbins=2) for axis in ['y']]
        plt.gca().set_xscale('log')
        plt.xlabel('Learning rate $\eta$')
        plt.ylabel(y)
        #ylims.append(plt.gca().get_ylim())
        plt_legend(ncol=1, prop={'size': text_size}, loc=(1.05, 0.1))
        plt.tight_layout()
        #tit = ''
        #plt.title('SF vs gradient rules')
        plt.savefig('./figures/fig3c_MSE_grad.pdf',
                    dpi=300,
                    bbox_inches='tight')
        plt.show(), plt.close()

    # FigS2: z and z2 (adapted from above but reduced) jjjj
    elif fig == 'figS2':

        folders = [
            'fig2_dim', 'fig2_beta', 'fig2_dim_pf', 'fig2_beta_pf', 'fig2_eta'
        ]
        folders2df(folders)

        # take dim or beta as x-axis, plot z or z2
        for selector, y in it.product([0, 1], ['z', 'z2']):
            tau_s = 0.1
            ntau_s = 0.1 if tau_s == 1 else 1
            x, meta = (['dim', 'beta0'], ['beta0', 'dim'])[selector]
            metas = [5 if 'std' in errbars else 6] if meta == 'dim' else [1]
            rules = ['exp', 'corr', 'pf_corr', 'exp_smooth', 'corr_smooth']
            out = out.sort_index()

            # meta to marker
            m2m = {0: 'o', 1: 'x', 2: 'v'}

            # Plot
            plt.ylabel(m2l[y])
            xrg = []
            for i, r in it.product(np.arange(len(metas)), rules):
                # fix dim = 10,11,12
                m = metas[i]
                out2 = out.loc[(out.rule == r) & (out[meta] == m) &
                               (out.tau_s != ntau_s)].sort_values(x)
                xyplt = out2.groupby(x).mean().reset_index(inplace=False)
                xyplt_err = out2.groupby(x).sem().reset_index(inplace=False)

                if len(xrg) == 0:
                    xrg = [xyplt[x].min(), xyplt[x].max()]
                else:
                    xrg = [
                        min(xrg[0], xyplt[x].min()),
                        max(xrg[1], xyplt[x].max())
                    ]

                plt.errorbar(xyplt[x],
                             xyplt[y],
                             xyplt_err[y],
                             color=r2c[r],
                             marker=m2m[i],
                             lw=2,
                             linestyle=r2s[r],
                             label=r2l[r])

            plt.plot(xrg,
                     np.ones(2) * (y == 'z2'),
                     '--',
                     c='gray',
                     lw=2,
                     label='exact')

            plt.title('{0}={1}'.format(x2l[meta], m))

            [plt.locator_params(axis=axis, nbins=2) for axis in ['x', 'y']]
            if x == 'dim':
                plt.gca().set_xticks([1, 15])
            elif x == 'beta0':
                plt.gca().set_xticks([0, 1])
                if y == 'z':
                    plt.gca().set_yticks([-0.1, 0, 0.1])
            plt.xlabel(x2l[x])

            if y == 'z2':
                plt.ylim([None, 2.5])
            if y == 'z':
                plt.ylim([-0.1, 0.075])
            plt_legend(ncol=2, prop={'size': text_size})
            plt.savefig('./figures/' + 'figS2_{1}_x{0}_m{2}_{3}.pdf'.format(
                x, y, m, 'errbars' if errbars != None else ''),
                        dpi=300,
                        bbox_inches='tight')
            plt.show(), plt.close()

    elif ('figS3' in fig) or (fig == 'fig2d') or (fig == 'fig2e'):
        r2l = {
            'exp': 'Diag. Syn. Fil.',  # 'DSF',
            'corr': 'Syn. Fil.',  #'SF',
            'exp_smooth': 'Diag. Samp. Syn. Fil.',  # 'DSF',
            'corr_smooth': 'Samp. Syn. Fil.',  # 'DSF',
            'exp_sample': r'$D(\Sigma), u^s$',
            'corr_sample': r'$\Sigma, u^s$',
            'grad': 'Gradient',
            'pf_corr': 'PF'
        }
        r2c = {'exp': 'k', 'corr': 'r', 'grad': 'gray'}

        median = False
        lognormal_error = False
        ql, qh = 0.40, 0.60
        difference = True
        joint_error = True
        plt_polynomial_fit = 'figS3' in fig  # fig
        plt_gmap = True
        rules = ['corr', 'exp']
        errorbar = plt.errorbar  #if maxed else plt.errorbar

        folders = ['fig2d'] if fig in ('fig2d', 'figS3a') else ['fig2e']
        out = folders2df(folders)

        if difference:
            ylab = r'log likelihood $\mathcal{L}_{\rm{SF}} - \mathcal{L}_{\rm{Grad}}$'
            ylab = r'Bayes ratio $\log \frac{p(\mathcal{D}|\mathcal{M}_{\rm{SF}})}{p(\mathcal{D}|\mathcal{M}_{\rm{Grad}})}$'
        else:
            ylab = r'rel. log likelihood $(\mathcal{L}-\mathcal{L}_0)/\mathcal{L}_g$'

        # incremental to total LL: LL = N*dLL
        N = 1 / mp['dt'] * mp['steps'] * mp['tau_ou']
        cmap = plt.cm.coolwarm

        if lognormal_error:  # error computed under log normal assumption
            out['log_z2d'] = np.log(out.z2d)
            out['log_z2d_pt'] = np.log(out.z2d_pt)
            metric = 'log_z2d'
        else:
            metric = 'z2d'
        ## GRADIENT
        lrs = out.loc[out.rule == 'grad', 'lr'].unique()
        cs = cmap(np.linspace(0, 1, len(lrs)))
        gg = out[(out.rule == 'grad')].drop(['sim', 'rule'],
                                            axis=1).groupby(['dim', 'lr'])
        if median:
            M = gg.median() * N
            S_low = gg.quantile(ql) * N
            S_high = gg.quantile(qh) * N
        else:
            M = gg.mean()
            S = gg.sem()

            if lognormal_error:
                Var = gg.var()
                M_ln = np.exp(M + Var / 2)
                S_ln = np.sqrt(M_ln**2 * (np.exp(Var) - 1))
                trials = Var / S**2
                # output
                M = M_ln
                S = S_ln / trials**0.5
            M, S = M * N, S * N

        # for each dim, sort by z2d and retain best values for fit
        irg = 3  # idx_range
        LL_max = []
        LL_sem = []
        for d in out.dim.unique():
            # get data around maximum
            imax = np.where(M.loc[d].index == M.loc[d][metric].idxmax())[0][0]
            left_i, right_i = max(imax - irg, 0), min(imax + irg + 1,
                                                      len(M.loc[d]))
            Mfit = M.loc[d].iloc[left_i:right_i]

            # fit 3rd order ploynomial
            X, Y = np.array(Mfit.index), Mfit[metric].values
            if len(X) < 7:
                print('')
                print('length X', len(X))
                print('')

            ff = Fit(X, Y, logx=True)
            ff.fit(fct=lambda x, th: th[0] + x * th[1] + x**2 * th[2] + x**3 *
                   th[3],
                   th0=np.array([Mfit[metric].mean(), 0, -1, 0]))
            # load and plt
            if (plt_polynomial_fit == False) or (d not in (3, 8)):
                ff.plt(do_plot=False)
            else:
                plt.scatter(X, Y, color='k', label='Simulations'), ff.plt(
                    c='red',
                    label='Polynomial fit'), plt.gca().set_xscale('log')
                plt.title(r'Learning rate vs loglikelihood $d = $' + str(d))
                plt.xlabel(r'Learning rate $\eta$')
                plt.ylabel(r'Loglikelihood')
                #plt.gca().set_yticks([3000, 4500])
                #plt.ylim([200, 600])
                plt.gca().set_yticks([3000, 4000])
                plt.ylim([2500, 4500])
                plt.xlim([0.03, 1.2])
                plt_legend()
                plt.savefig('./figures/figS3ab_example_fit_{0}.pdf'.format(d),
                            dpi=300,
                            bbox_inches='tight')
                plt.show(), plt.close()

            LL_max.append(ff.ylin.max())

            # get error bars
            location = ff.ylin.argmax()
            ll_sem = []
            if median:
                for S_ in [S_low, S_high]:
                    # select
                    s_ = S_.loc[d].iloc[left_i:right_i][metric].values
                    # fit
                    ff_err = Fit(X, s_, logx=True)
                    ff_err.fit(fct=lambda x, th: th[0] + x * th[1] + x**2 * th[
                        2] + x**3 * th[3],
                               th0=np.array([s_.mean(), 0, -1, 0]))
                    ff_err.plt(do_plot=False)
                    ll_sem.append(ff_err.ylin[location])
                # extract
                LL_sem.append(ll_sem)
            else:
                s_ = S.loc[d].iloc[left_i:right_i][metric].values
                # fit
                ff_err = Fit(X, s_, logx=True)
                ff_err.fit(fct=lambda x, th: th[0] + x * th[1] + x**2 * th[2] +
                           x**3 * th[3],
                           th0=np.array([s_.mean(), 0, -1, 0]))
                ff_err.plt(do_plot=False)
                LL_sem.append(ff_err.ylin[location])

        if plt_polynomial_fit:  # stop fct here if figS3 was requested
            return ()

        # plot
        cgrad = 'k' if len(rules) == 1 else 'gray'
        if median:
            LL_max, LL_sem = np.array(LL_max), np.array(LL_sem).T
            LL_sem_low, LL_sem_high = LL_sem[0], LL_sem[1]
            grad_pts = LL_max * (difference == False)
            plt.plot(out.dim.unique(),
                     grad_pts,
                     marker='o',
                     label='Gradient',
                     c=cgrad,
                     lw=2)
            plt.fill_between(out.dim.unique(),
                             LL_sem_low - LL_max * difference,
                             LL_sem_high - LL_max * difference,
                             color=cgrad,
                             alpha=0.3)

        else:
            LL_max, LL_sem = np.array(LL_max), np.array(LL_sem)
            grad_pts = LL_max * (difference == False)
            plt_errorbar(out.dim.unique(),
                         grad_pts,
                         LL_sem * (joint_error == False),
                         c=cgrad,
                         lw=2,
                         marker='o',
                         label='Gradient')

        ## BAYESIAN
        for r in rules:
            out2 = out[out.rule == r]
            dims = out2.dim.unique()
            gg = out2.drop(['sim', 'rule'], axis=1).groupby(['dim'])
            xplt = gg.mean().index
            if median:
                yplt = (gg.median()[metric] * N - LL_max * difference)
                yerr_l = gg.quantile(ql)[metric] * N - LL_max
                yerr_h = gg.quantile(qh)[metric] * N - LL_max
                yerr = (np.vstack([(yplt - yerr_l).values,
                                   (yerr_h - yplt).values]))
                plt.errorbar(xplt,
                             yplt,
                             yerr,
                             marker='o',
                             label=r2l[r] + ' BR',
                             color=r2c[r])
                if plt_gmap:
                    yplt = (gg.median()[metric + '_pt'] * N - LL_max)
                    yerr_l = gg.quantile(ql)[metric + '_pt'] * N - LL_max
                    yerr_h = gg.quantile(qh)[metric + '_pt'] * N - LL_max
                    yerr = (np.vstack([(yplt - yerr_l).values,
                                       (yerr_h - yplt).values]))
                    plt.errorbar(xplt,
                                 yplt,
                                 yerr,
                                 ls='--',
                                 marker='o',
                                 label=r2l[r] + ' MAP',
                                 color=r2c[r])
            else:
                M = gg.mean()
                S = gg.sem()

                if lognormal_error:
                    Var = gg.var()
                    M_ln = np.exp(M + Var / 2)
                    S_ln = np.sqrt(M_ln**2 * (np.exp(Var) - 1))
                    trials = Var / S**2
                    # output
                    M = M_ln
                    S = S_ln / trials**0.5
                M, S = M * N, S * N

                yplt = M[metric] - LL_max * difference
                yerr = (S[metric]**2 +
                        LL_sem**2)**0.5 if joint_error else S[metric]
                errorbar(xplt,
                         yplt,
                         yerr,
                         marker='o',
                         label=r2l[r] + ' BR',
                         color=r2c[r])
                if plt_gmap:
                    yplt = M[metric + '_pt'] - LL_max * difference
                    yerr = (S[metric + '_pt']**2 +
                            LL_sem**2)**0.5 if joint_error else S[metric +
                                                                  '_pt']
                    plt.errorbar(xplt,
                                 yplt,
                                 yerr,
                                 marker='o',
                                 ls='--',
                                 label=r2l[r] + ' MAP',
                                 color=r2c[r])

        if fig in 'fig2e':
            # ground truth dimension: plot bar
            dimgm = out2['dim-gm'].unique()
            plt.plot([dimgm, dimgm], plt.gca().get_ylim(), '--k')
            plt.gca().set_xticks([dims.min(), dimgm[0], dims.max()])
            xlab = r'Model dimension $d$'
            tit = 'Model mismatch'
            plt.ylim(
                [-250 if difference else None, 200 if difference else None])
        else:
            plt.gca().set_xticks([dims.min(), dims.max()])
            xlab = r'Dimension $d$'
            tit = 'Standard setting'
            #plt.ylim([-100,None])
            plt.ylim(
                [-250 if difference else None, 200 if difference else None])
        order = [1, 2, 0, 3, 4] if plt_gmap else [1, 2, 0]
        plt_legend(ncol=2, loc=3, prop={'size': 13}, order=order)
        plt.xlabel(xlab), plt.ylabel(ylab)
        [plt.locator_params(axis=axis, nbins=3) for axis in ['y']]
        plt.title(tit)
        #plt.ylim([None if difference else 0,None])
        plt.savefig('./figures/{0}_M{1}_{2}{3}.pdf'.format(
            fig, mp['M'], plt_gmap, 'ln' if lognormal_error else ''),
                    dpi=300,
                    bbox_inches='tight')
        plt.show(), plt.close()
