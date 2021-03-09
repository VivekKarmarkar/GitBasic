# -*- coding: utf-8 -*-
"""
Created on Sun Mar  7 20:49:13 2021

@author: vkarmarkar
"""

import numpy as np
import matplotlib.pyplot as plt

n_max = 9
t = np.linspace(-2,4,100);
x_pos = np.arange(4)
amplitude = []
phase = []
bars = [str(k) + 'f' if k>1 else 'f' for k in range(1, n_max, 2)]

a_0 = 0.5
def a_k(k):
    a_k_val = (2/(k*np.pi)**2) * (np.cos(k*np.pi) - 1)
    return a_k_val

sum_k = a_0
for j in range(1, n_max, 2):
    #k = j+1
    k = j
    m = int((j+1)/2)
    coeff_k = a_k(k)
    phase_k = np.nan
    if coeff_k < 0:
        phase_k = np.pi
    else:
        phase_k = 0
    phase.append(phase_k)
    term_k = coeff_k*np.cos(k*np.pi*t)
    sum_k += term_k
    amplitude.append(abs(coeff_k))
    plt.figure()
    plt.plot(t, sum_k, 'r', label='Sum')
    plt.plot(t, term_k, 'b', label='Term')
    plt.axhline(y=0, c='k', linestyle='--')
    plt.axvline(x=0, c='k')
    plt.grid()
    plt.xlabel('t')
    plt.ylabel('f(t)')
    plt.title('Plot for sum of first {} non zero terms and the term no {}'.format(m, m))
    plt.legend(loc='upper right', )
    plt.ylim([-0.5, 1])
    plt.show()

plt.figure()
plt.bar(x_pos, amplitude)
plt.xticks(x_pos, tuple(bars))
plt.yticks([2/(np.pi**2), 4/(np.pi**2)], ('2/(pi**2)', '4/(pi**2)'))
plt.title('Amplitude Spectra')
plt.show()

plt.figure()
plt.bar(x_pos, phase)
plt.xticks(x_pos, tuple(bars))
plt.yticks([np.pi/2, np.pi], ('pi/2', 'pi'))
plt.title('Phase Spectra')
plt.show()
