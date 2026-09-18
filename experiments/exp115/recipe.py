"""Frozen scientific design for the four-state, white-noise PING closure."""

from itertools import product

SLUG = "exp115"


def configuration():
    return {
        "schema": "exp115.recipe/v3",
        "cells": {
            "E": {"tau_m_ms": 20.0, "g_L_uS": 0.05, "tau_ref_ms": 1.2},
            "I": {"tau_m_ms": 5.0, "g_L_uS": 0.10, "tau_ref_ms": 0.6},
        },
        "rest_mV": -65.0,
        "reset_mV": -65.0,
        "threshold_mV": -50.0,
        "tau_AMPA_ms": 2.0,
        "G_EI_uS": 1.0,
        "G_IE_uS": 2.0,
        "dV_exc_mV": 65.0,
        "dV_inh_mV": 15.0,
        "tau_GABA_grid_ms": [4.5, 6.0, 9.0, 12.0, 18.0, 27.0],
        "sigma_grid_mV": [3.0, 4.0, 5.0, 6.0],
        "kappa_grid": [0.5, 0.75, 1.0, 1.5, 2.0],
        "reference": {"tau_GABA_ms": 6.0, "sigma_mV": 4.0, "kappa": 1.0},
        "drive_interval_nA": [0.0, 4.0],
        "drive_counts": [401, 801, 1601],
        "initial_rates_per_ms": [0.005, 0.002],
        "equilibrium_reuse": "within_run_across_kappa_with_flow_residual_recheck",
        "quad_abs": 1e-12,
        "quad_rel": 1e-10,
        "quad_limit": 200,
        "brent_abs_nA": 1e-10,
        "brent_rel": 1e-12,
        "equilibrium_residual": 1e-10,
        "tightening_factor": 100.0,
        "transversality_steps_nA": [1e-4, 5e-5, 2.5e-5],
        "criticality_ramp": {
            "tau_GABA_ms": [4.5, 6.0, 9.0, 12.0, 18.0, 27.0],
            "sigma_mV": 4.0,
            "kappa": 1.0,
            "span_nA": [-0.1, 0.55],
            "points": 25,
            "duration_ms": 2000.0,
            "observation_start_ms": 1500.0,
            "observation_step_ms": 1.0,
            "initial_rate_perturbation_per_ms": 1e-3,
            "solver": "LSODA",
            "rtol": 1e-7,
            "atol": 1e-10,
            "max_step_ms": 1.0,
            "amplitude_threshold_per_ms": 1e-4,
            "branch_gap_threshold_per_ms": 1e-4,
            "minimum_amplitude_squared_r2": 0.9,
        },
        "criteria": {
            "critical_real_scaled": 1e-9,
            "noncritical_real_scaled": 1e-8,
            "eigen_separation_scaled": 1e-8,
            "hopf_identity": 1e-9,
            "frequency_Hz": 1e-4,
            "transversality_rel": 1e-3,
            "drive_convergence_nA": 1e-7,
            "sign_margin": 10.0,
        },
    }


def validate(cfg):
    if cfg != configuration():
        raise ValueError("unsupported or inconsistent exp115 scientific recipe")
    return cfg


def conditions(cfg):
    return [
        {"tau_GABA_ms": tau, "sigma_mV": sigma, "kappa": kappa}
        for tau, sigma, kappa in product(
            cfg["tau_GABA_grid_ms"], cfg["sigma_grid_mV"], cfg["kappa_grid"]
        )
    ]


def condition_id(condition):
    return "tau-{:g}--sigma-{:g}--kappa-{:g}".format(
        condition["tau_GABA_ms"], condition["sigma_mV"], condition["kappa"]
    ).replace(".", "p")


def requires_criticality_ramp(condition, cfg):
    ramp = cfg["criticality_ramp"]
    return (
        condition["tau_GABA_ms"] in ramp["tau_GABA_ms"]
        and condition["sigma_mV"] == ramp["sigma_mV"]
        and condition["kappa"] == ramp["kappa"]
    )
