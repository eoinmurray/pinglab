
from .population_isi_cv import isi_cv_per_neuron
import numpy as np


def test_isi_cv_per_neuron():
    print("Testing isi_cv_per_neuron...")
    rng = np.random.default_rng(0)

    class DummySpikes:
        def __init__(self, times, ids):
            self.times = times
            self.ids = ids

    # 100 neurons, gamma/Poisson ISIs
    N_neurons = 100
    spike_times = []
    spike_ids = []

    for nid in range(N_neurons):
        rate = 10.0  # Hz
        T = 1.0      # seconds
        t = 0.0
        times_n = []
        while t < T:
            # exponential ISI
            isi = rng.exponential(1.0 / rate)
            t += isi
            if t < T:
                times_n.append(t * 1000.0)  # ms
        spike_times.extend(times_n)
        spike_ids.extend([nid] * len(times_n))

    spikes = DummySpikes(np.array(spike_times), np.array(spike_ids))
    ids_out, cv = isi_cv_per_neuron(spikes, min_spikes=5, ddof=0)

    median_cv = np.median(cv)

    assert len(ids_out) > 0
    assert len(ids_out) == len(cv)
    assert median_cv > 0.8 and median_cv < 1.0

    print("median CV (should be ~1):", median_cv)
