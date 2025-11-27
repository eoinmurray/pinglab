from pinglab.types import InstrumentsResults
import matplotlib.pyplot as plt
from pathlib import Path
from pinglab.plots.styles import save_both

def save_instrument_traces(instruments: InstrumentsResults, save_path):
    save_path = Path(save_path)

    if instruments.V is not None:
        times = instruments.times # x axis
        neuron_ids = instruments.neuron_ids # label
        types = instruments.types # color
        V = instruments.V # y axis, line per neuron

        def plot_voltage():
            plt.figure(figsize=(8, 8))
            for i, neuron_id in enumerate(neuron_ids):
                neuron_type = types[i] if types is not None else 'unknown'
                label = f'Neuron {neuron_id} ({"E" if neuron_type == 0 else "I"})'
                plt.plot(times, V[:, i], label=label)
            plt.xlabel('Time (ms)')
            plt.ylabel('Membrane Potential (mV)')
            plt.title('Instrumented Neuron Voltage Traces')
            plt.legend()
            plt.tight_layout()

        save_both(save_path / 'trace_voltage.png', plot_voltage)


    if instruments.g_e is not None:
        times = instruments.times
        neuron_ids = instruments.neuron_ids
        types = instruments.types
        g_e = instruments.g_e

        def plot_g_e():
            plt.figure(figsize=(8, 8))
            for i, neuron_id in enumerate(neuron_ids):
                neuron_type = types[i] if types is not None else 'unknown'
                label = f'Neuron {neuron_id} ({"E" if neuron_type == 0 else "I"})'
                plt.plot(times, g_e[:, i], label=label)
            plt.xlabel('Time (ms)')
            plt.ylabel('Excitatory Conductance (nS)')
            plt.title('Instrumented Neuron Excitatory Conductance Traces')
            plt.legend()
            plt.tight_layout()

        save_both(save_path / 'trace_g_e.png', plot_g_e)

    if instruments.g_i is not None:
        times = instruments.times
        neuron_ids = instruments.neuron_ids
        types = instruments.types
        g_i = instruments.g_i

        def plot_g_i():
            plt.figure(figsize=(8, 8))
            for i, neuron_id in enumerate(neuron_ids):
                neuron_type = types[i] if types is not None else 'unknown'
                label = f'Neuron {neuron_id} ({"E" if neuron_type == 0 else "I"})'
                plt.plot(times, g_i[:, i], label=label)
            plt.xlabel('Time (ms)')
            plt.ylabel('Inhibitory Conductance (nS)')
            plt.title('Instrumented Neuron Inhibitory Conductance Traces')
            plt.legend()
            plt.tight_layout()

        save_both(save_path / 'trace_g_i.png', plot_g_i)

    # Population mean plots
    if instruments.V_mean_E is not None or instruments.V_mean_I is not None:
        times = instruments.times

        def plot_voltage_mean():
            plt.figure(figsize=(8, 8))
            if instruments.V_mean_E is not None:
                plt.plot(times, instruments.V_mean_E, label='E population mean', linewidth=2)
            if instruments.V_mean_I is not None:
                plt.plot(times, instruments.V_mean_I, label='I population mean', linewidth=2)
            plt.xlabel('Time (ms)')
            plt.ylabel('Mean Membrane Potential (mV)')
            plt.title('Population Mean Voltage Traces')
            plt.legend()
            plt.tight_layout()

        save_both(save_path / 'trace_voltage_mean.png', plot_voltage_mean)

    if instruments.g_e_mean_E is not None or instruments.g_e_mean_I is not None:
        times = instruments.times

        def plot_g_e_mean():
            plt.figure(figsize=(8, 8))
            if instruments.g_e_mean_E is not None:
                plt.plot(times, instruments.g_e_mean_E, label='E population mean', linewidth=2)
            if instruments.g_e_mean_I is not None:
                plt.plot(times, instruments.g_e_mean_I, label='I population mean', linewidth=2)
            plt.xlabel('Time (ms)')
            plt.ylabel('Mean Excitatory Conductance (nS)')
            plt.title('Population Mean Excitatory Conductance')
            plt.legend()
            plt.tight_layout()

        save_both(save_path / 'trace_g_e_mean.png', plot_g_e_mean)

    if instruments.g_i_mean_E is not None or instruments.g_i_mean_I is not None:
        times = instruments.times

        def plot_g_i_mean():
            plt.figure(figsize=(8, 8))
            if instruments.g_i_mean_E is not None:
                plt.plot(times, instruments.g_i_mean_E, label='E population mean', linewidth=2)
            if instruments.g_i_mean_I is not None:
                plt.plot(times, instruments.g_i_mean_I, label='I population mean', linewidth=2)
            plt.xlabel('Time (ms)')
            plt.ylabel('Mean Inhibitory Conductance (nS)')
            plt.title('Population Mean Inhibitory Conductance')
            plt.legend()
            plt.tight_layout()

        save_both(save_path / 'trace_g_i_mean.png', plot_g_i_mean)