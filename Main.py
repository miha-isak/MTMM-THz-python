import os
import json
import numpy as np
import matplotlib.pyplot as plt
import scipy.io

def time2freq(t_ref, E_ref, t_sam, E_sam, minTHz, maxTHz):
    # Align to zero
    t_ref = t_ref - np.min(t_ref)
    t_sam = t_sam - np.min(t_sam)

    # Plot time-domain signals
    plt.figure()
    plt.plot(t_ref, E_ref, linewidth=3, label='E_{Reference}')
    plt.plot(t_sam, E_sam, linewidth=3, label='E_{Reference}')
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
    t_min = min(np.min(t_ref), np.min(t_sam))
    t_max = max(np.max(t_ref), np.max(t_sam))
    num_points = max(len(t_ref), len(t_sam))
    time = np.linspace(t_min, t_max, num_points)

    # Frequency bounds (Hz)
    f_min = minTHz * 1e12
    f_max = maxTHz * 1e12

    # Pulse maxima
    idx_ref = np.argmax(E_ref)
    idx_sam = np.argmax(E_sam)
    t0r = time[idx_ref]
    t0s = time[idx_sam]
    dtpeaks = t0s - t0r
    dtmin = dtpeaks
    dT = t_max - t_min

    # FFT parameters
    N = len(time)
    dt = time[1] - time[0]
    Fs = 1 / dt
    N_pad = N  # pad_factor = 1

    E_ref_padded = np.concatenate([E_ref, np.zeros(N_pad - N)])
    E_sam_padded = np.concatenate([E_sam, np.zeros(N_pad - N)])

    f_full = Fs * np.arange(0, N_pad // 2 + 1) / N_pad
    omega_full = 2 * np.pi * f_full

    E_ref_fft = np.fft.fft(E_ref_padded)
    E_sam_fft = np.fft.fft(E_sam_padded)

    E_ref = E_ref_fft[:len(f_full)]
    E_sam = E_sam_fft[:len(f_full)]

    # Frequency filter
    mask = (f_full >= f_min) & (f_full <= f_max)
    f = f_full[mask]
    omega = omega_full[mask]
    E_ref = E_ref[mask]
    E_sam = E_sam[mask]

    # Reduced phase
    phi0_ref = omega * t0r
    phi0_sam = omega * t0s
    phi_red_ref = np.angle(E_ref * np.exp(-1j * phi0_ref))
    phi_red_sam = np.angle(E_sam * np.exp(-1j * phi0_sam))

    # Unwrapped phase difference
    delta_phi_star_0 = np.unwrap(phi_red_sam - phi_red_ref)

    # Linear fit to center region
    center_fraction = 0.5
    N_center = int(np.round(len(f) * center_fraction))
    start_idx = int(np.round((len(f) - N_center) / 2))
    center_idx = np.arange(start_idx, start_idx + N_center)

    omega_center = omega[center_idx]
    delta_phi_center = delta_phi_star_0[center_idx]
    p = np.polyfit(omega_center, delta_phi_center, 1)
    B = p[1]
    delta_phi_0 = delta_phi_star_0 - 2 * np.pi * np.round(B / (2 * np.pi))

    # Final corrected phase
    phi_offset = 0
    delta_phi = -1 * (delta_phi_0 - phi0_ref + phi0_sam + phi_offset)

    # Plot FFT magnitude
    plt.figure()
    plt.plot(f, np.log10(np.abs(E_ref)), linewidth=3, label='E_{Reference}')
    plt.plot(f, np.log10(np.abs(E_sam)), linewidth=3, label='E_{Sample}')
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

    scipy.io.savemat('Test.mat', {
        'EsovEr': EsovEr,
        'f': f,
        'E_sam': E_sam,
        'E_ref': E_ref,
        'delta_phi': delta_phi,
        'lambda0': lambda0,
        'dtpeaks': dtpeaks,
        't0s': t0s,
        't0r': t0r,
        'dtmin': dtmin,
        'dT': dT
    })


def TM_DBR_test1(d0):
    mat = scipy.io.loadmat(r'MTMM Python\Test copy.mat')
    lambda0 = np.array(mat['lambda0']).flatten()
    L = lambda0.size

    ns = np.array(d0[0:L], dtype=np.float64).flatten()
    ksmp = np.array(d0[L:2*L], dtype=np.float64).flatten()
    ns = ns + 1j * ksmp  # Ensure shape (L,) complex

    t_smpl = float(d0[2*L])  # Ensure scalar
    theta0 = 0

    d = np.array(mat['d']).flatten()
    idx = np.isnan(d)
    d[idx] = t_smpl

    flag = 0
    dlimit = np.array(mat['dlimit']).flatten()
    nk = np.array(mat['nk']).flatten()
    nr = float(mat['nr'].flatten()[0])

    t_cs2 = MTMM(d, lambda0, theta0, nr, ns, flag, dlimit, nk)
    print(idx)


def MTMM(d, lambda0, theta0, nr, ns, flag, dlimit, nk):
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

        if isinstance(n_s, np.ndarray):
            n_s = n_s.item()  # Ensure scalar

        #!!!
        if isinstance(n_s, np.ndarray):
            if n_s.size == 1:
                n_s = n_s.item()
            else:
                raise ValueError(f"n_s is not scalar, shape: {n_s.shape}")
        #Construct full refractive index profile for this lambda
        n=nk.copy()
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
                D[0,1]=0
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

# === Setup ===
path = os.getcwd() #current working directory
os.chdir('MTMM Python') #Change the current working directory
os.chdir('Data') #Change the current working directory

# === Structural Info ===
samplename = 'SiO2'
reference = np.loadtxt(f'{samplename}_Reference.txt')
sample = np.loadtxt(f'{samplename}_Sample.txt')
pop_size = 100
maxit = 1000

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

time2freq(t_file_ref, E_file_ref, t_file_sig, E_file_sig, minTHz, maxTHz)

# === Load Data and Analytical n,k Extraction ===
mat = scipy.io.loadmat(r'MTMM Python\Test copy.mat')
print(mat.keys())
EsovEr = mat['EsovEr'].flatten()
f = mat['f'].flatten()
delta_phi = mat['delta_phi'].flatten()
dtpeaks = mat['dtpeaks'].flatten()[0]
dT = mat['dT'].flatten()[0]

c = 299792458
L = np.array(mat['lambda0']).size
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

scipy.io.savemat('Test.mat', {
    **mat,
    'n_anltic': np.array(n_anltic),
    'k_anltic': np.array(k_anltic)
})

n_anltic = np.array(n_anltic)
k_anltic = np.array(k_anltic)

nhalf = np.array([(n_anltic[0,:] + n_anltic[1,:]) / 2,
         (n_anltic[0,:] + n_anltic[2,:]) / 2])
khalf = np.array([(k_anltic[0,:] + k_anltic[1,:]) / 2,
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
TM_DBR_test1(lb)
print(nvars,L)
plot_opts = {'linestyle': ':', 'marker': 'o', 'linewidth': 1.6}
axis_opts = {'FontSize':14, 'FontWeight':'bold', 'LineWidth': 1.5}
fig_opts = {'Units':'Inches', 'Position':[1, 1, 6, 4]}

# Plotting the real part (n_anltic)
plt.figure()
plt.plot(f, n_anltic[0, :], label='n (real part)', linewidth=7)
plt.plot(f, nhalf[0], label='n (real 1 part)', linewidth=3)
plt.plot(f, k_anltic[0, :], label='k (imaginary part)', linewidth=3)
plt.xlabel('Frequency (Hz)', fontsize=16, fontweight='bold', fontname='Cambria')
plt.ylabel('Refractive Index', fontsize=16, fontweight='bold', fontname='Cambria')
plt.xticks(fontsize=12, fontweight='bold', fontname='Cambria')
plt.yticks(fontsize=12, fontweight='bold', fontname='Cambria')
plt.legend(prop={'weight':'bold', 'family':'Cambria'})
plt.grid(False)  # turn off grid
plt.tight_layout()
plt.show()
