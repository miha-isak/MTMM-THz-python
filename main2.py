import os
import json
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import numpy.typing as npt
from scipy.optimize import differential_evolution
import os

def time2freq(t_ref:npt.NDArray, E_ref:npt.NDArray, t_sam:npt.NDArray, E_sam:npt.NDArray, minTHz:float, maxTHz:float):
    '''


    t_ref, E_ref: Time & electric field for reference signal.
    t_sam, E_sam: Same for sample.
    minTHz, maxTHz: Frequency bounds(in Tera Hertz) to analyze.
    return:
    f: frequency vector.
    delta_phi: corrected phase difference.
    dT: total scan duration
    lambda0: wavelengths.

    dtpeaks: time shift between peaks.
    E_ref, E_sam: FFT of signals in frequency domain.
    '''
    # Align to zero
    t_ref:npt.NDArray = t_ref - np.min(t_ref)
    t_sam:npt.NDArray = t_sam - np.min(t_sam)

    # Plot time-domain signals
    plt.figure()
    plt.plot(t_ref, E_ref, linewidth=3, label='E_Reference')
    plt.plot(t_sam, E_sam, linewidth=3, label='E_Sample')
    plt.legend(prop={'weight':'bold', 'family':'Cambria'})
    plt.grid(False)  # turn off grid
    # plt.tight_layout()
    # plt.show()
    plt.xlabel('Time (sec)', fontsize=16, fontweight='bold', fontname='Cambria')
    plt.ylabel('Electric field intensity (a.u.)', fontsize=16, fontweight='bold', fontname='Cambria')
    plt.xticks(fontsize=12, fontweight='bold', fontname='Cambria')
    plt.yticks(fontsize=12, fontweight='bold', fontname='Cambria')
    plt.legend()


    # Common time grid
    t_min:float = min(np.min(t_ref), np.min(t_sam))
    t_max:float = max(np.max(t_ref), np.max(t_sam))
    num_points:int = max(len(t_ref), len(t_sam))
    time:npt.NDArray = np.linspace(t_min, t_max, num_points)
    # Frequency bounds (Hz)
    f_min:float = minTHz * 1e12
    f_max:float = maxTHz * 1e12

    # Pulse maxima
    idx_ref:np.int64 = np.argmax(E_ref)
    idx_sam:np.int64 = np.argmax(E_sam)
    t0r:float = time[idx_ref]
    t0s:float = time[idx_sam]
    dtpeaks:float = t0s - t0r
    dtmin:float = dtpeaks
    dT:float = t_max - t_min

    # FFT parameters
    N:int = len(time)
    
    dt:float = time[1] - time[0]
    '''time step'''
    Fs:float = 1 / dt
    N_pad:int = N  # pad_factor = 1

    E_ref_padded:npt.NDArray = np.concatenate([E_ref, np.zeros(N_pad - N)])
    E_sam_padded:npt.NDArray = np.concatenate([E_sam, np.zeros(N_pad - N)])

    f_full:npt.NDArray = Fs * np.arange(0, N_pad // 2 + 1) / N_pad
    omega_full:npt.NDArray = 2 * np.pi * f_full

    E_ref_fft:npt.NDArray[np.complex128] = np.fft.fft(E_ref_padded)
    E_sam_fft:npt.NDArray[np.complex128] = np.fft.fft(E_sam_padded)

    E_ref:npt.NDArray[np.complex128] = E_ref_fft[:len(f_full)]
    E_sam:npt.NDArray[np.complex128] = E_sam_fft[:len(f_full)]

    # Frequency filter
    mask:npt.NDArray[np.bool] = (f_full >= f_min) & (f_full <= f_max)
    f:npt.NDArray = f_full[mask]
    omega:npt.NDArray = omega_full[mask]
    E_ref:npt.NDArray = E_ref[mask]
    E_sam:npt.NDArray = E_sam[mask]

    # Reduced phase
    phi0_ref:npt.NDArray = omega * t0r
    phi0_sam:npt.NDArray = omega * t0s
    phi_red_ref:npt.NDArray = np.angle(E_ref * np.exp(-1j * phi0_ref))
    phi_red_sam:npt.NDArray = np.angle(E_sam * np.exp(-1j * phi0_sam))

    # Unwrapped phase difference
    delta_phi_star_0:npt.NDArray = np.unwrap(phi_red_sam - phi_red_ref)

    # Linear fit to center region
    center_fraction:float = 0.5
    N_center:int = int(np.round(len(f) * center_fraction))
    start_idx:int = int(np.round((len(f) - N_center) / 2))
    center_idx:npt.NDArray[np.int64] = np.arange(start_idx, start_idx + N_center)

    omega_center:npt.NDArray = omega[center_idx]
    delta_phi_center:npt.NDArray = delta_phi_star_0[center_idx]
    p:npt.NDArray = np.polyfit(omega_center, delta_phi_center, 1)
    B:np.float64 = p[1]
    delta_phi_0:npt.NDArray[np.float64] = delta_phi_star_0 - 2 * np.pi * np.round(B / (2 * np.pi))

    # Final corrected phase
    phi_offset:float = 0
    delta_phi:npt.NDArray = -1 * (delta_phi_0 - phi0_ref + phi0_sam + phi_offset)

    # Plot FFT magnitude
    plt.figure()
    plt.plot(f, np.log10(np.abs(E_ref)), linewidth=3, label='E_Reference')
    plt.plot(f, np.log10(np.abs(E_sam)), linewidth=3, label='E_Sample')
    #plt.title('One-sided Fourier Transform')
    plt.xlabel('Frequency (Hz)', fontsize=16, fontweight='bold', fontname='Cambria')
    plt.ylabel('Electric field intensity (a.u.)', fontsize=16, fontweight='bold', fontname='Cambria')
    plt.xticks(fontsize=12, fontweight='bold', fontname='Cambria')
    plt.yticks(fontsize=12, fontweight='bold', fontname='Cambria')
    plt.legend()
    plt.grid(False)

    # Plot phase difference
    plt.figure()
    plt.plot(f * 1e-12, delta_phi, 'k', linewidth=1.5)
    plt.xlabel('Frequency (THz)')
    plt.ylabel('Phase Difference (rad)')
    plt.title('Corrected Phase Difference')
    plt.grid(True)

    # Save data
    c = 299792458  # speed of light (m/s)
    f = f[::-1]
    EsovEr = (E_sam / E_ref)[::-1]
    E_sam = E_sam[::-1]
    E_ref = E_ref[::-1]
    delta_phi = delta_phi[::-1]
    lambda0 = (c / f) * 1e9

    # scipy.io.savemat(r'Test.mat', {
    #     'EsovEr': EsovEr,
    #     'f': f,
    #     'E_sam': E_sam,
    #     'E_ref': E_ref,
    #     'delta_phi': delta_phi,
    #     'lambda0': lambda0,
    #     'dtpeaks': dtpeaks,
    #     't0s': t0s,
    #     't0r': t0r,
    #     'dtmin': dtmin,
    #     'dT': dT
    # })
    return {
        'EsovEr': EsovEr,
        'f': f,
        # 'E_sam': E_sam,
        # 'E_ref': E_ref,
        'delta_phi': delta_phi,
        'lambda0': lambda0,
        'dtpeaks': dtpeaks,
        # 't0s': t0s,
        # 't0r': t0r,
        # 'dtmin': dtmin,
        'dT': dT
    }


def TM_DBR_test1(d0,lambda0:npt.NDArray,EsovEr:npt.NDArray,d:npt.NDArray,dlimit:npt.NDArray,nk:npt.NDArray,nr:float):
    # mat = scipy.io.loadmat(r'Test.mat')
    # lambda0 = mat['lambda0'].flatten()
    # EsovEr = mat['EsovEr'].flatten()
    # d = mat['d'].flatten()
    # dlimit = mat['dlimit'].flatten()
    # nk = mat['nk'].flatten()
    # nr = float(mat['nr'].flatten()[0])

    L = len(lambda0)
    ns = d0[:L] + 1j * d0[L:2*L]
    t_smpl = float(d0[2*L])

    idx = np.isnan(d)
    d[idx] = t_smpl

    t_cs_ref = MTMM(d, lambda0, 0, nr, ns, 0, dlimit, nk)
    t_cs_sam = MTMM(d, lambda0, 0, nr, ns, 1, dlimit, nk)

    deviations = np.abs(EsovEr - (t_cs_sam / t_cs_ref))
    return np.sum(deviations)


def MTMM(d:npt.NDArray, lambda0:npt.NDArray, theta0:int, nr:float, ns:npt.NDArray, flag:int, dlimit:npt.NDArray, nk:npt.NDArray):
    N=d.size
    t_cs = np.zeros(len(lambda0), dtype=complex)

    idx=np.isnan(nk)

    for a in range(lambda0.size):
        k0=2*np.pi/lambda0[a]
        #Assign refractive index of sample or reference
        if flag:
            n_s = ns[a]
        else:
            n_s = nr

        # if isinstance(n_s, npt.NDArray):
        #     n_s = n_s.item()  # Ensure scalar

        # #!!!
        # if isinstance(n_s, npt.NDArray):
        #     if n_s.size == 1:
        #         n_s = n_s.item()
        #     else:
        #         raise ValueError(f"n_s is not scalar, shape: {n_s.shape}")
        #Construct full refractive index profile for this lambda
        n = nk.astype(np.complex128).copy()
        n[idx]=n_s

        M_sPol=np.eye(2,dtype=complex)

        for c in range(N-1):
            k_x=n[c]*k0
            phi=k_x*d[c]

            D = np.array([
                [(n[c] + n[c+1]) / (2 * n[c]), (n[c] - n[c+1]) / (2 * n[c])],
                [(n[c] - n[c+1]) / (2 * n[c]), (n[c] + n[c+1]) / (2 * n[c])]
            ], dtype=complex)

            if d[c+1]>dlimit[c+1]:
                D[:,1]=0
            #Propagation matrix
            if c==0:
                P=np.eye(2,dtype=complex)
            else:
                P = np.array([
                    [np.exp(1j * phi), 0],
                    [0, np.exp(-1j * phi)]
                ], dtype=complex)
            M_sPol = M_sPol @ P @ D

        #Transmission coefficient
        t_cs[a] = 1 / M_sPol[0, 0]
    return t_cs


if __name__ == '__main__':
    # === Setup ===
    path = os.getcwd() #current working directory
    os.chdir('Data') #Change the current working directory

    # === Structural Info ===
    samplename = 'H2O'
    reference = np.loadtxt(f'{samplename}_Reference.txt')
    sample = np.loadtxt(f'{samplename}_Sample.txt')
    pop_size = 1
    maxit = 1000
    c = 299792458

    with open(f'{samplename}.json', 'r') as f:
        data = json.load(f)

    nr = data["settings"]["calibration_index"]
    minTHz = data["settings"]["minTHz"]
    maxTHz = data["settings"]["maxTHz"]
    d = [layer["d_nm"] for layer in data["Sample"]]
    nk = np.array([layer["n"] for layer in data["Sample"]])
    t_smpl0 = d[np.where(np.isnan(nk))[0][0]]

    # === Time-domain to Frequency-domain ===
    os.chdir(path)
    t_file_ref = reference[:, 0] * 1e-12
    E_file_ref = reference[:, 1]
    t_file_sig = sample[:, 0] * 1e-12
    E_file_sig = sample[:, 1]
    

    # mat = scipy.io.loadmat(r'source.mat')
    tmp=time2freq(t_file_ref, E_file_ref, t_file_sig, E_file_sig, minTHz, maxTHz)
    # mat = scipy.io.loadmat(r'Test.mat')
    # === Load Data and Analytical n,k Extraction ===
    EsovEr = tmp['EsovEr'].flatten()
    f = tmp['f'].flatten()
    delta_phi = tmp['delta_phi'].flatten()
    dtpeaks = tmp['dtpeaks'].flatten()[0]
    dT = tmp['dT'].flatten()[0]
    lambda0=tmp['lambda0']
    L = np.array(lambda0).size
    nn0 = 1 + c * dtpeaks / (t_smpl0 * 1e-9)
    neff = nk.copy()
    neff[np.isnan(neff)] = nn0
    dlimit = (c * dT / (2 * neff)) * 1e9

    PH = np.arange(6)
    delta_phi_values = 2 * np.pi * np.floor((PH + 1) / 2) * (-1) ** PH

    n_anltic = []
    k_anltic = []

    for delta_add in delta_phi_values:
        delta_phi2 = delta_phi + delta_add
        n_anlt = nr + c * delta_phi2 / (2 * np.pi * f * t_smpl0 * 1e-9)
        constnt = (4 * n_anlt * nr) / ((np.abs(EsovEr) * (n_anlt + nr)**2))
        k_anlt = (c / (2 * np.pi * f * t_smpl0 * 1e-9)) * np.log(np.abs(constnt))
        n_anltic.append(n_anlt)
        k_anltic.append(k_anlt)

    # scipy.io.savemat(r'Test.mat', {
    #     **mat,
    #     'n_anltic': np.array(n_anltic),
    #     'k_anltic': np.array(k_anltic)
    # })

    n_anltic = np.array(n_anltic)
    k_anltic = np.array(k_anltic)

    nhalf = np.array([(n_anltic[0,:] + n_anltic[1,:]) / 2,
            (n_anltic[0,:] + n_anltic[2,:]) / 2])
    khalf = -np.array([(k_anltic[0,:] + k_anltic[1,:]) / 2,
            (k_anltic[0,:] + k_anltic[2,:]) / 2])
    lb = np.concatenate([
        np.min(np.real(nhalf),axis=0),                                # scalar → 1D
        -2 * np.max(np.abs(k_anltic)) * np.ones(L),              # already 1D
        [t_smpl0]                                                 # scalar → 1D
    ]).flatten()
    ub = np.concatenate([
        np.max(np.real(nhalf),axis=0),                                # scalar → 1D
        np.zeros(L),              # already 1D
        [t_smpl0]                                                 # scalar → 1D
    ]).flatten()
    nvars = lb.size

    initial_candidate = np.concatenate([n_anltic[0], -k_anltic[0], [t_smpl0]])
    initial_pop = initial_candidate#np.tile(initial_candidate, (pop_size, 1))

    bounds = list(zip(lb, ub))
    args=(
        lambda0.flatten(),
        EsovEr.flatten(),
        np.array(d).flatten(),
        dlimit.flatten(),
        nk.flatten(),
        float(nr))
    result = differential_evolution(
        func=TM_DBR_test1,
        args=args,
        bounds=bounds,
        maxiter=maxit,
        popsize=pop_size,
        x0=initial_pop,
        polish=False,
        disp=True,
        workers=1,
    )
    d0_opt = result.x
    plot_opts = {'linestyle': ':', 'marker': 'o', 'linewidth': 1.6}
    axis_opts = {'FontSize':14, 'FontWeight':'bold', 'LineWidth': 1.5}
    fig_opts = {'Units':'Inches', 'Position':[1, 1, 6, 4]}
    n=d0_opt[:L]
    k=-d0_opt[L:2*L]
    # Plotting the real part (n_anltic)
    if not os.path.exists('result'):
        os.mkdir('result')
    scipy.io.savemat(f'result/{samplename}_Results.mat',{'d0':d0_opt})
    plt.figure()
    plt.plot(f, n, label='n (real part)', **plot_opts)
    plt.plot(f, k, label='k (imaginary part)', **plot_opts)
    plt.xlabel('Frequency (Hz)', fontsize=16, fontweight='bold', fontname='Cambria')
    plt.ylabel('Refractive Index', fontsize=16, fontweight='bold', fontname='Cambria')
    plt.xticks(fontsize=12, fontweight='bold', fontname='Cambria')
    plt.yticks(fontsize=12, fontweight='bold', fontname='Cambria')
    plt.legend(prop={'weight':'bold', 'family':'Cambria'})
    plt.grid(False)  # turn off grid
    plt.tight_layout()
    plt.show()