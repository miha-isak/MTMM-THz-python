import scipy
import numpy as np

m_python=scipy.io.loadmat(r"D:\CCNY\MTMM Python\result\H2O_Results.mat")
m_matlab=scipy.io.loadmat(r"D:\CCNY\MTMM-THz-TDS\Results\H2O_Results.mat")
d0_python=m_python['d0']
d0_matlab=m_matlab['d0']
delta=(d0_matlab-d0_python)
error=np.sum(delta*delta)
print(error)