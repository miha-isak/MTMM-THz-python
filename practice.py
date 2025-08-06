import os
path=os.getcwd()
print(os.getcwd())
x=os.chdir("Data")
print(os.getcwd())

import numpy as np
H2O=np.loadtxt("H2O_Sample.txt")
t=H2O[:,0]*1e-12
Sig=H2O[:,1]
N = len(t)
dt = t[1] - t[0]         # Time step
Fs = 1 / dt              # Sampling frequency
SigF = np.fft.fft(Sig)   # Full FFT
fF = np.fft.fftfreq(N, d=dt)[:N // 2]  # Frequency axis (positive only)
SigF1 = SigF[:N // 2]  # One-sided FFT

import matplotlib.pyplot as plt
'''plt.figure
plt.plot(t,Sig)'''
plt.figure()
plt.plot(fF,np.abs(SigF1))
plt.figure()
plt.plot(fF,np.unwrap(np.angle(SigF1)))
plt.xlabel("F")
plt.ylabel("phase")
plt.show()



import json
os.chdir(path)
with open("testjson.json","r") as tj:
    setupinfo=json.load(tj)
    print(setupinfo)