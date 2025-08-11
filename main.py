import os
import json
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import numpy.typing as npt
from scipy.optimize._differentialevolution import DifferentialEvolutionSolver
from scipy.optimize import _differentialevolution
from scipy import optimize
import os
import warnings
import tqdm

def time2freq(t_ref:npt.NDArray,
              E_ref:npt.NDArray,
              t_sam:npt.NDArray,
              E_sam:npt.NDArray,
              minTHz:float,
              maxTHz:float)->dict[str,np.ndarray|float]:
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
    t_ref-=np.min(t_ref)
    t_sam-=np.min(t_sam)

    # Plot time-domain signals
    plt.figure()
    plt.plot(t_ref, E_ref, linewidth=3, label='E_Reference')
    plt.plot(t_sam, E_sam, linewidth=3, label='E_Sample')
    plt.legend(prop={'weight':'bold', 'family':'Cambria'})
    plt.grid(True)
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
    minHz:float = minTHz * 1e12
    maxHz:float = maxTHz * 1e12

    # Pulse maxima
    idx_ref:np.int64 = np.argmax(E_ref)
    idx_sam:np.int64 = np.argmax(E_sam)
    t0r:float = time[idx_ref]
    t0s:float = time[idx_sam]
    dtpeaks:float = t0s - t0r
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

    E_ref_from_fft:npt.NDArray[np.complex128] = E_ref_fft[:len(f_full)]
    E_sam_from_fft:npt.NDArray[np.complex128] = E_sam_fft[:len(f_full)]

    # Frequency filter
    mask:npt.NDArray[np.bool] = (f_full >= minHz) & (f_full <= maxHz)
    f:npt.NDArray = f_full[mask]
    omega:npt.NDArray = omega_full[mask]
    E_ref_from_fft = E_ref_from_fft[mask]
    E_sam_from_fft = E_sam_from_fft[mask]

    # Reduced phase
    phi0_ref:npt.NDArray = omega * t0r
    phi0_sam:npt.NDArray = omega * t0s
    phi_red_ref:npt.NDArray = np.angle(E_ref_from_fft * np.exp(-1j * phi0_ref))
    phi_red_sam:npt.NDArray = np.angle(E_sam_from_fft * np.exp(-1j * phi0_sam))

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
    b:np.float64 = p[1]
    delta_phi_0:npt.NDArray[np.float64] = delta_phi_star_0 - 2 * np.pi * np.round(b / (2 * np.pi))

    # Final corrected phase
    phi_offset:float = 0
    delta_phi:npt.NDArray = -1 * (delta_phi_0 - phi0_ref + phi0_sam + phi_offset)

    # Plot FFT magnitude
    plt.figure()
    plt.plot(f, np.log10(np.abs(E_ref_from_fft)), linewidth=3, label='E_Reference')
    plt.plot(f, np.log10(np.abs(E_sam_from_fft)), linewidth=3, label='E_Sample')
    #plt.title('One-sided Fourier Transform')
    plt.xlabel('Frequency (Hz)', fontsize=16, fontweight='bold', fontname='Cambria')
    plt.ylabel('Electric field intensity (a.u.)', fontsize=16, fontweight='bold', fontname='Cambria')
    plt.xticks(fontsize=12, fontweight='bold', fontname='Cambria')
    plt.yticks(fontsize=12, fontweight='bold', fontname='Cambria')
    plt.legend()
    plt.grid(True)

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
    EsovEr = (E_sam_from_fft / E_ref_from_fft)[::-1]
    E_sam_from_fft = E_sam_from_fft[::-1]
    E_ref_from_fft = E_ref_from_fft[::-1]
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


def TM_DBR_test1(d0:npt.NDArray,
                 lambda0:npt.NDArray,
                 EsovEr:npt.NDArray,
                 d:npt.NDArray,
                 dlimit:npt.NDArray,
                 nk:npt.NDArray,
                 nr:float):
    # mat = scipy.io.loadmat(r'Test.mat')
    # lambda0 = mat['lambda0'].flatten()
    # EsovEr = mat['EsovEr'].flatten()
    # d = mat['d'].flatten()
    # dlimit = mat['dlimit'].flatten()
    # nk = mat['nk'].flatten()
    # nr = float(mat['nr'].flatten()[0])

    l = len(lambda0)
    ns = d0[:l] + 1j * d0[l:2*l]
    t_smpl = float(d0[2*l])

    idx = np.isnan(d)
    d[idx] = t_smpl

    t_cs_ref = MTMM(d, lambda0, 0, nr, ns, 0, dlimit, nk)
    t_cs_sam = MTMM(d, lambda0, 0, nr, ns, 1, dlimit, nk)

    deviations = np.abs(EsovEr - (t_cs_sam / t_cs_ref))
    return np.sum(deviations)


def MTMM(d:npt.NDArray,
         lambda0:npt.NDArray,
         theta0:int,
         nr:float,
         ns:npt.NDArray,
         flag:int,
         dlimit:npt.NDArray,
         nk:npt.NDArray):
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


class Solver(DifferentialEvolutionSolver):
   def solve(self):
        """
        Runs the DifferentialEvolutionSolver.

        Returns
        -------
        res : OptimizeResult
            The optimization result represented as a `OptimizeResult` object.
            Important attributes are: ``x`` the solution array, ``success`` a
            Boolean flag indicating if the optimizer exited successfully,
            ``message`` which describes the cause of the termination,
            ``population`` the solution vectors present in the population, and
            ``population_energies`` the value of the objective function for
            each entry in ``population``.
            See `OptimizeResult` for a description of other attributes. If
            `polish` was employed, and a lower minimum was obtained by the
            polishing, then OptimizeResult also contains the ``jac`` attribute.
            If the eventual solution does not satisfy the applied constraints
            ``success`` will be `False`.
        """
        nit, warning_flag = 0, False
        status_message = _differentialevolution._status_message['success']

        # The population may have just been initialized (all entries are
        # np.inf). If it has you have to calculate the initial energies.
        # Although this is also done in the evolve generator it's possible
        # that someone can set maxiter=0, at which point we still want the
        # initial energies to be calculated (the following loop isn't run).
        if np.all(np.isinf(self.population_energies)):
            self.feasible, self.constraint_violation = (
                self._calculate_population_feasibilities(self.population))

            # only work out population energies for feasible solutions
            self.population_energies[self.feasible] = (
                self._calculate_population_energies(
                    self.population[self.feasible]))

            self._promote_lowest_energy()

        # do the optimization.
        progres_bar=hasattr(self,'progres_bar') and self.progres_bar
        with tqdm.tqdm(total=self.maxiter, unit="it",disable=not progres_bar) as pbar:
            for nit in range(1, self.maxiter + 1):
                # evolve the population by a generation
                try:
                    next(self)
                except StopIteration:
                    warning_flag = True
                    if self._nfev > self.maxfun:
                        status_message = _differentialevolution._status_message['maxfev']
                    elif self._nfev == self.maxfun:
                        status_message = ('Maximum number of function evaluations'
                                        ' has been reached.')
                    break

                if self.disp:
                    print(f"differential_evolution step {nit}: f(x)="
                        f" {self.population_energies[0]}",
                        f"average {np.average(self.population_energies)}"
                        )
                if progres_bar:
                    pbar.update(1)
                    pbar.set_postfix({'f(x)':f'{self.population_energies[0]:.3e}','average':f'{np.average(self.population_energies):.3e}'})
                if self.callback:
                    c = self.tol / (self.convergence + _differentialevolution._MACHEPS)
                    res = self._result(nit=nit, message="in progress")
                    res.convergence = c
                    try:
                        warning_flag = bool(self.callback(res))
                    except StopIteration:
                        warning_flag = True

                    if warning_flag:
                        status_message = 'callback function requested stop early'

                # should the solver terminate?
                if warning_flag or self.converged():
                    break

            else:
                status_message = _differentialevolution._status_message['maxiter']
                warning_flag = True

        DE_result = self._result(
            nit=nit, message=status_message, warning_flag=warning_flag
        )

        if self.polish and not np.all(self.integrality):
            # can't polish if all the parameters are integers
            if np.any(self.integrality):
                # set the lower/upper bounds equal so that any integrality
                # constraints work.
                limits, integrality = self.limits, self.integrality
                limits[0, integrality] = DE_result.x[integrality]
                limits[1, integrality] = DE_result.x[integrality]

            polish_method = 'L-BFGS-B'

            if self._wrapped_constraints:
                polish_method = 'trust-constr'

                constr_violation = self._constraint_violation_fn(DE_result.x)
                if np.any(constr_violation > 0.):
                    warnings.warn("differential evolution didn't find a "
                                  "solution satisfying the constraints, "
                                  "attempting to polish from the least "
                                  "infeasible solution",
                                  UserWarning, stacklevel=2)
            if self.disp:
                print(f"Polishing solution with '{polish_method}'")
            result = optimize.minimize(self.func,
                              np.copy(DE_result.x),
                              method=polish_method,
                              bounds=self.limits.T,
                              constraints=self.constraints)

            self._nfev += result.nfev
            DE_result.nfev = self._nfev

            # Polishing solution is only accepted if there is an improvement in
            # cost function, the polishing was successful and the solution lies
            # within the bounds.
            if (result.fun < DE_result.fun and
                    result.success and
                    np.all(result.x <= self.limits[1]) and
                    np.all(self.limits[0] <= result.x)):
                DE_result.fun = result.fun
                DE_result.x = result.x
                DE_result.jac = result.jac
                # to keep internal state consistent
                self.population_energies[0] = result.fun
                self.population[0] = self._unscale_parameters(result.x)

        if self._wrapped_constraints:
            DE_result.constr = [c.violation(DE_result.x) for
                                c in self._wrapped_constraints]
            DE_result.constr_violation = np.max(
                np.concatenate(DE_result.constr))
            DE_result.maxcv = DE_result.constr_violation
            if DE_result.maxcv > 0:
                # if the result is infeasible then success must be False
                DE_result.success = False
                DE_result.message = ("The solution does not satisfy the "
                                     f"constraints, MAXCV = {DE_result.maxcv}")

        return DE_result


def differential_evolution(func, bounds, args=(), strategy='best1bin',
                           maxiter=1000, popsize=15, tol=0.01,
                           mutation=(0.5, 1), recombination=0.7, seed=None,
                           callback=None, disp=False,progres_bar:bool=False, polish=True,
                           init='latinhypercube', atol=0, updating='immediate',
                           workers=1, constraints=(), x0=None, *,
                           integrality=None, vectorized=False):
    rng=np.random.default_rng(seed)
    with Solver(func, bounds, args=args,
                                     strategy=strategy,
                                     maxiter=maxiter,
                                     popsize=popsize, tol=tol,
                                     mutation=mutation,
                                     recombination=recombination,
                                     rng=rng, polish=polish,
                                     callback=callback,
                                     disp=disp, init=init, atol=atol,
                                     updating=updating,
                                     workers=workers,
                                     constraints=constraints,
                                     x0=x0,
                                     integrality=integrality,
                                     vectorized=vectorized) as solver:
        solver.progres_bar=progres_bar
        ret = solver.solve()

    return ret


def main(sample_name:str,pop_size:int = 2,maxit:int = 1000):
    # === Structural Info ===
    reference:npt.NDArray[np.floating] = np.loadtxt(rf'Data/{sample_name}_Reference.txt')
    sample:npt.NDArray[np.floating]  = np.loadtxt(rf'Data/{sample_name}_Sample.txt')
    c:float = 299792458#speed of light

    with open(fr'Data/{sample_name}.json', 'r') as file:
        data:dict = json.load(file)

    nr:float = float(data["settings"]["calibration_index"])
    minTHz:float = float(data["settings"]["minTHz"])
    maxTHz:float = float(data["settings"]["maxTHz"])
    d:npt.NDArray[np.floating] = np.array([layer["d_nm"] for layer in data["Sample"]])
    nk:npt.NDArray[np.floating] = np.array([layer["n"] for layer in data["Sample"]])
    t_smpl0 = float(d[np.where(np.isnan(nk))[0][0]])

    # === Time-domain to Frequency-domain ===
    t_file_ref = reference[:, 0] * 1e-12
    E_file_ref = reference[:, 1]
    t_file_sig = sample[:, 0] * 1e-12
    E_file_sig = sample[:, 1]
    


    tmp:dict[str,npt.NDArray|float] = time2freq(t_file_ref, E_file_ref, t_file_sig, E_file_sig, minTHz, maxTHz)

    # === Load Data and Analytical n,k Extraction ===
    EsovEr:npt.NDArray[np.floating]    = tmp['EsovEr'].flatten()
    f:npt.NDArray[np.floating]         = tmp['f'].flatten()
    delta_phi:npt.NDArray[np.floating] = tmp['delta_phi'].flatten()
    dtpeaks:float   = tmp['dtpeaks'].flatten()[0]
    dT:float        = tmp['dT'].flatten()[0]
    lambda0:npt.NDArray[np.floating]   = tmp['lambda0']
    l:int = lambda0.size
    nn0:float = 1 + c * dtpeaks / (t_smpl0 * 1e-9)
    neff = nk.copy()
    neff[np.isnan(neff)] = nn0
    dlimit = (c * dT / (2 * neff)) * 1e9

    ph = np.arange(6)
    delta_phi_values = 2 * np.pi * np.floor((ph + 1) / 2) * (-1) ** ph

    n_anltic_list:list[npt.NDArray[np.complexfloating]] = []
    k_anltic_list:list[npt.NDArray[np.complexfloating]] = []

    for delta_add in delta_phi_values:
        delta_add:float
        delta_phi2 = delta_phi + delta_add
        n_anlt = (nr + c * delta_phi2 / (2 * np.pi * f * t_smpl0 * 1e-9)).astype(complex)
        constnt = (4 * n_anlt * nr) / ((np.abs(EsovEr) * (n_anlt + nr)**2))
        k_anlt = (c / (2 * np.pi * f * t_smpl0 * 1e-9)) * np.log(constnt.astype(complex))
        n_anltic_list.append(n_anlt)
        k_anltic_list.append(k_anlt)

    # scipy.io.savemat(r'Test.mat', {
    #     **mat,
    #     'n_anltic': np.array(n_anltic),
    #     'k_anltic': np.array(k_anltic)
    # })


    n_anltic = np.array(n_anltic_list)
    k_anltic = np.array(k_anltic_list)

    nhalf:npt.NDArray[np.complexfloating] = np.array([(n_anltic[0] + n_anltic[1]) / 2,
            (n_anltic[0] + n_anltic[2]) / 2])
    
    lb:npt.NDArray[np.floating] = np.concatenate([
        np.min(np.real(nhalf),axis=0),                                # scalar → 1D
        -2 * np.max(np.abs(k_anltic)) * np.ones(l),              # already 1D
        [t_smpl0]                                                 # scalar → 1D
    ]).flatten()
    ub:npt.NDArray[np.floating] = np.concatenate([
        np.max(np.real(nhalf),axis=0),                                # scalar → 1D
        np.zeros(l),              # already 1D
        [t_smpl0]                                                 # scalar → 1D
    ]).flatten()
    initial_pop:npt.NDArray[np.floating] = np.concatenate([np.real(n_anltic[0]), np.real(-k_anltic[0]), [t_smpl0]])
    np.clip(initial_pop,lb,ub,initial_pop)
    bounds = list(zip(lb, ub))
    args=(
        lambda0.flatten(),
        EsovEr.flatten(),
        d.flatten(),
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
        tol=0.0001,
        polish=False,
        disp=False,
        progres_bar=True,
        workers=-1,
        updating='deferred'
    )
    print(result.message)
    d0_opt = result.x
    plot_opts = {'linestyle': ':', 'marker': 'o', 'linewidth': 1.6}
    n=d0_opt[:l]
    k=-d0_opt[l:2*l]
    # Plotting the real part (n_anltic)
    if not os.path.exists('result'):
        os.mkdir('result')
    scipy.io.savemat(f'result/{sample_name}_Results.mat',{'d0':d0_opt})

    fig, ax = plt.subplots(figsize=(6, 5))

    ax.plot(f, n,color='blue', label="n")
    ax.set_xlabel('Frequency (THz)', fontsize=12, fontweight='bold', fontname='Arial')
    ax.set_ylabel('Refractive index, n', fontsize=12, fontweight='bold', fontname='Arial')
    ax.tick_params(axis='both', labelsize=12)
    plt.legend(loc="upper right")
    for spine in ax.spines.values():
        spine.set_linewidth(1)

    ax_right = ax.twinx()
    ax_right.plot(f, k,color='orange',label="k")
    ax_right.set_ylabel('Extinction coefficient, k', fontsize=12, fontweight='bold', fontname='Arial')
    ax_right.tick_params(axis='both', labelsize=12)
    for spine in ax_right.spines.values():
        spine.set_linewidth(1)

    plt.tight_layout()
    ax.grid(True, axis='both')
    ax_right.grid(False)
    lines_ax, labels_ax = ax.get_legend_handles_labels()
    lines_right, labels_right = ax_right.get_legend_handles_labels()
    ax.legend(lines_ax + lines_right, labels_ax + labels_right, loc="upper right")
    plt.savefig(f'result/{sample_name}_Results.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    main(sample_name='PTFE',
        pop_size=4,
        maxit=2000)
#TODO: element titles