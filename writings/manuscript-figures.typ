// Descriptions without numerical claims: safe before data exist and shared by
// the gallery and comparison without importing another article's constructed body.
#let comparison-figures = (
  (experiment: "exp025", filename: "results_compound.png", title: "Trained PING versus COBA", description: [Compare training curves, rasters, and the accuracy–rate frontier.]),
  (experiment: "exp038", filename: "loop_transfer_compound.png", title: "Inference-time loop activation", description: [Compare firing rates and accuracy as the inhibitory loop is enabled at inference.]),
  (experiment: "exp049", filename: "training_curves.svg", title: "Released-loop training", description: [Compare frozen recurrence with trainable recurrent initializations.]),
  (experiment: "exp041", filename: "rate_vs_fgamma.svg", title: "Rate versus gamma frequency", description: [Compare firing rate, gamma frequency, and accuracy across inhibitory decay settings.]),
  (experiment: "exp046", filename: "spikes_per_cycle_distribution.svg", title: "Spikes per gamma cycle", description: [Compare the distribution of excitatory spikes per cell and gamma cycle.]),
  (experiment: "exp037", filename: "perturbation_curves.svg", title: "Spike perturbation robustness", description: [Compare accuracy under spike deletion and added spikes.]),
  (experiment: "exp042", filename: "rhythm_compound.png", title: "Inhibitory timing perturbations", description: [Compare cell-wise jitter with shifts of intact inhibitory volleys.]),
  (experiment: "exp044", filename: "dt_sweep.svg", title: "Integration-timestep comparison", description: [Compare firing rate and accuracy across integration timesteps.]),
)

#let figure-description(experiment) = {
  let matches = comparison-figures.filter(item => item.experiment == experiment)
  if matches.len() > 0 { matches.first().description } else { [Selected experiment figure.] }
}
