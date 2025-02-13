# -*- coding: utf-8 -*-
"""
Created on Sat Sep 30 09:15:56 2023

@author: Hamza
"""

import numpy as np
import matplotlib.pyplot as plt
import emcee
import corner
import pandas as pd

# Constants
H = 6.62607015e-34  # Planck's constant (J s)
C = 299792458  # Speed of light (m/s)
K = 1.380649e-23  # Boltzmann constant (J/K)

# Frequency domain for plotting
FREQUENCY_DOMAIN = np.geomspace(0.1, 1000, 850)  # Log-spaced for better visualization


def load_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load COBE (FIRAS) and ARCADE 2 data."""
    cobe_data = np.loadtxt(
        'lambda.gsfc.nasa.gov_data_cobe_firas_monopole_spec_firas_monopole_spec_v1.txt', unpack=True
    )
    arcade_data = np.loadtxt('Intesidades_ARCADE_ERROR.txt', unpack=True)
    
    return (
        cobe_data[0] * 30,  # Convert cm^-1 to GHz  (Frequency)
        cobe_data[1] * np.pi,  # Convert MJy/sr to MJy (Energy)
        cobe_data[3] * np.pi * 1e-3,  # Convert kJy/sr to MJy (Error of energy)
        arcade_data[0], #Frequency
        arcade_data[1], #Energy
        arcade_data[2] #Error of energy
    )


def blackbody_model(theta: float, nu: np.ndarray) -> np.ndarray:
    """Compute the Planck blackbody spectrum."""
    nu_hz = nu * 1e9  # GHz to Hz
    exponent = (H * nu_hz) / (K * theta)
    return (2 * np.pi * H * nu_hz**3 / C**2) / (np.expm1(exponent)) / 1e-20  # Convert to MJy


def log_likelihood(theta: float, nu: np.ndarray, spectrum: np.ndarray, error_spectrum: np.ndarray) -> float:
    """Compute the log-likelihood."""
    model_spectrum = blackbody_model(theta, nu)
    return -0.5 * np.sum(((spectrum - model_spectrum) / error_spectrum) ** 2)


def log_prior(theta: float) -> float:
    """Define a uniform prior for temperature."""
    return 0.0 if 0.0 < theta < 5.0 else -np.inf


def log_probability(theta: float, nu: np.ndarray, spectrum: np.ndarray, error_spectrum: np.ndarray) -> float:
    """Compute the log-posterior probability."""
    lp = log_prior(theta)
    return lp + log_likelihood(theta, nu, spectrum, error_spectrum) if np.isfinite(lp) else -np.inf


def run_mcmc(p0: np.ndarray, nwalkers: int, niter: int, ndim: int, log_prob, data: tuple) -> emcee.EnsembleSampler:
    """Run the MCMC sampler."""
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob, args=data)
    sampler.run_mcmc(p0, niter, progress=True)
    return sampler


def plot_results(sampler: emcee.EnsembleSampler, nu: np.ndarray, spectrum: np.ndarray, error_spectrum: np.ndarray, label: str):
    """Plot the MCMC results with color differentiation."""
    colors = {'ARCADE': 'red', 'COBE': 'blue', 'ARCADE_COBE': 'purple'}
    samples = sampler.get_chain(discard=150, flat=True)[:, 0]
    plt.figure(figsize=(12, 8))
    plt.errorbar(nu, spectrum, yerr=error_spectrum, fmt='o', label=label, color=colors.get(label, 'black'))
    for theta in np.random.choice(samples, 500):
        plt.plot(FREQUENCY_DOMAIN, blackbody_model(theta, FREQUENCY_DOMAIN), 'k', alpha=0.1)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("$\\nu$ [GHz]")
    plt.ylabel("$I_{\\nu}$ [MJy]")
    plt.title(f'Blackbody Fit for {label}')
    plt.legend()
    plt.show()
    corner.corner(samples[:, None], labels=['T'], show_titles=True, quantiles=[0.5])
    plt.show()


def save_results(sampler: emcee.EnsembleSampler, label: str, nwalkers: int, niter: int, nu: np.ndarray, spectrum: np.ndarray, error_spectrum: np.ndarray):
    """Save MCMC results."""
    samples = sampler.get_chain(discard=150, flat=True)[:, 0]
    results = {
        label: {
            'frecuencia': nu.tolist(),
            'energia': spectrum.tolist(),
            'error': error_spectrum.tolist(),
            'muestra': samples.tolist()
        },
        'walkers': nwalkers,
        'modelo': "Planck's Law",
        'pasos': niter
    }
    pd.DataFrame(results).T.to_json(f'walkers_{nwalkers}_pasos_{niter}_MCMC_{label}.json', orient='records')


def main():
    """Main function to execute MCMC analysis."""
    frecuencia_COBE, espectro_COBE, error_COBE, frecuencia_ARCADE, espectro_ARCADE, error_ARCADE = load_data()
    frecuencia_ambos = np.concatenate((frecuencia_ARCADE, frecuencia_COBE))
    espectro_ambos = np.concatenate((espectro_ARCADE, espectro_COBE))
    error_ambos = np.concatenate((error_ARCADE, error_COBE))
    datasets = {
        'ARCADE & COBE': (frecuencia_ambos, espectro_ambos, error_ambos),
        'COBE': (frecuencia_COBE, espectro_COBE, error_COBE),
        'ARCADE': (frecuencia_ARCADE, espectro_ARCADE, error_ARCADE)
    }
    
    nwalkers, niter = 20, 1000
    initial = np.array([4.9])
    p0 = initial + 1e-3 * np.random.randn(nwalkers, len(initial))
    
    for label, data in datasets.items():
        sampler = run_mcmc(p0, nwalkers, niter, len(initial), log_probability, data)
        plot_results(sampler, *data, label)
        # save_results(sampler, label, nwalkers, niter, *data)


if __name__ == "__main__":
    main()