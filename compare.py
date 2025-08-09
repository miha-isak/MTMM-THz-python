import scipy
import numpy as np
sample_name='H2O'
m_python=scipy.io.loadmat(fr"D:\CCNY\MTMM Python\result\{sample_name}_Results.mat")
m_matlab=scipy.io.loadmat(fr"D:\CCNY\MTMM-THz-TDS\Results\{sample_name}_Results.mat")
d0_python=m_python['d0']
d0_matlab=m_matlab['d0']
delta=(d0_matlab-d0_python)
error=np.sum(np.abs(delta))/delta.size
print(error,d0_python-d0_matlab)