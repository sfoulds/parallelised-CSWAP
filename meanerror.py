######################################################
#### Thank you Gerard Pelegri for the base code! #####
######################################################
import numpy as np
import scipy as scp
from scipy.optimize import minimize_scalar,minimize
from scipy.optimize import fsolve
from scipy.integrate import quad
from functools import partial
# from pairinteraction import picomplex as pi
import itertools
from sympy.utilities.iterables import multiset_permutations
# import rydsim
#import matplotlib.pyplot as plt
import copy
from qutip import *
#from CEfunctions import *
opts=Options()
opts.atol=1e-10
opts.rtol=1e-10
opts.nsteps=5000
opts.store_final_state=True
opts.store_states=True
opts.normalize_output=False

import matplotlib as mpl
from scipy.optimize import *

mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['mathtext.rm'] = 'serif'
mpl.rcParams['font.family'] = 'serif'
import matplotlib.pyplot as plt
import itertools

from scipy.stats import binom

textwidth=13 #cm
lw = 2.5
xl = r"$\alpha$"
effxl = r'$\alpha$'
yl = 'Mean error'
effyl = '$E(Y)$'
axislabel_fontsize = 21
ticksize = 8
ticklabel_fontsize = 18
legend_fontsize = 14
legendtitle_fontsize = 20
msize = 15
mwidth = 3
c = ['black', 'mediumvioletred', 'cornflowerblue', 'tomato', 'mediumseagreen', 'blue', 'rebeccapurple', 'black']
c2 = ['black', 'mediumseagreen', 'rebeccapurple', 'cornflowerblue', 'mediumvioletred']

def error_plot():
	plt.xlabel(xl,fontsize=axislabel_fontsize)
	plt.ylabel(yl,fontsize=axislabel_fontsize)
	plt.tick_params(axis="both",direction="in",size=ticksize/2,which="minor")
	plt.tick_params(axis="both",direction="in",size=ticksize,which="major")
	plt.yticks(fontsize=ticklabel_fontsize)
	plt.xticks(fontsize=ticklabel_fontsize)
	plt.legend(frameon=False)
	return

#######################

def ci_error_mean(N,p,ptrue,alpha):
    c1_mean=0
    c2_mean=0
    mean_error=0
    for k in range(N+1):
        c1,c2=binomial_ci(k, N, 1-alpha)
        c1_mean+=c1*binom.pmf(k,N,p)
        c2_mean+=c2*binom.pmf(k,N,p)
        est=k/N
        error=abs(est-ptrue)/ptrue
        mean_error+=binom.pmf(k,N,p)*error
    return c1_mean,c2_mean,mean_error
    
def binomial_ci(x, n, alpha):
    #x is number of successes, n is number of trials
    from scipy import stats
    if x==0:
        c1 = 0
    else:
        c1 = stats.beta.interval(1-alpha, x,n-x+1)[0]
    if x==n:
        c2=1
    else:
        c2 = stats.beta.interval(1-alpha, x+1,n-x)[1]
    return c1, c2

####################
colours = ['#882255', '#E27559', '#DDCC77', '#44AA99', '#88CCEE', '#332288']
lines = ['solid', 'dotted', 'solid', 'dashed', 'solid', 'dashdot']

def f2(data, x, a, b, c):
    ###### INPUT (IF FITTING): create fit 'f' to test
    f = 1-x*data**4
    return f

n=2
nalphas=50
alphas=np.linspace(0,2,nalphas) #1000
Nm=100

cs=[0.25, 0.5, 0.75, 1]
#Ps = ((1-np.exp(-4*alphas**2))**2) / 4
c1_mean_ps=np.zeros(len(alphas))
c2_mean_ps=np.zeros(len(alphas))
mean_error_ps=np.zeros(len(alphas))
mean_error_ps2=np.zeros(len(alphas))
for j, c in enumerate(cs):
    Ps = (c**2 * (1-np.exp(-4*alphas**2))**2) / 4
    for i in range(len(Ps)):
        c1_mean_ps[i],c2_mean_ps[i],mean_error_ps[i]=ci_error_mean(Nm,np.abs(Ps[i]),np.abs(Ps[i]),0.8)
    mean_error_ps2 =mean_error_ps/2
    plt.plot(alphas, mean_error_ps2, label=c, color=colours[j+1], linestyle=lines[j+1])
    #print(mean_error_ps2[1:])
    top=int(nalphas)
    
    try:
        #popt,pcov = curve_fit(f2, alphas[1:top], mean_error_ps2[1:top])  # IF FITTING, COMMENT BACK IN
        #print(popt)                  # IF FITTING, COMMENT BACK IN
        #plt.plot(alphas, f2(alphas, popt[0], popt[1], popt[2], popt[3]))
        pass
    except RuntimeError:
        print("Error - curve_fit failed")

error_plot()
plt.yscale('log')
plt.xticks([0, 0.5, 1, 1.5, 2])
legend = plt.legend(title='$C_2\'$', loc='upper right', fontsize=legend_fontsize, ncol=2)
plt.setp(legend.get_title(),fontsize=legendtitle_fontsize)
plt.savefig('current.pdf',bbox_inches="tight")
#plt.savefig('mixedMs2.pdf',bbox_inches="tight")
