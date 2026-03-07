// title: study.1-rhythmicity-index
// date: 2026-02-19
// description: Scanning EE weight parameters to map rhythmicity across a PING network.

#let config = json("_artifacts/study.1-rhythmicity-index/config.json")





*Readers note:* click images to make them larger, you can then use arrow keys
to scroll through them.


= Introduction


- We use an autocorrelation rhythmicity metric to characterise the AI-PING
  transition. 
- We scan over a range of mean and standard deviation values for the EE synaptic
weights, and plot:
  - rasters
  - population rates
  - autocorrelations
  - rhythmicity metric 
  - a heatmap of the autocorrelation rhythmicity metric
    across the EE mean/std parameter space.


== Network architecture


// Gallery: ping-simple.png
// Path: _assets/study.1-rhythmicity-index
// #figure(image("_assets/study.1-rhythmicity-index/ping-simple.png"), caption: [Simple PING architecture with coupling between E and I populations with E self connections.])


The network is setup as follows:

1. 400 E neurons.
2. 100 I neurons.
3. Noisy tonic input to E population only.
4. Weak EI coupling: $W_(ei)$ = $N$(#config.edges.at(2).w.mean, #config.edges.at(2).w.std).
5. Weak IE coupling: $W_(ie)$ = $N$(#config.edges.at(3).w.mean, #config.edges.at(3).w.std).
6. Variable EE coupling: $W_(ee)$ = $N$($mu$, $sigma$) where $mu$ and $sigma$ are scanned over a range of values.

We run two scans:

1. Scan over mean $W_(ee)$ with fixed std $W_(ee)$ of 0.
2. Scan over std $W_(ee)$ with fixed mean $W_(ee)$ of 0.


= Results



== Rasters


// Gallery: raster_mean_scan_2_00*.png, raster_mean_scan_2_02*.png, raster_mean_scan_2_06*.png, 
// #figure(image("_artifacts/study.1-rhythmicity-index/raster_mean_scan_2_00dark.png"), caption: [Rasters while scanning over $\mu_{ee}$ and keeping $\sigma_{ee}$
zeroed. Left: No PING rhythm in the E population. Middle: Weak PING rhythm
emerging. Right: Strong PING rhythm with clear bursts in the E population.])


// Gallery: raster_std_scan_1_00*.png, raster_std_scan_1_03*.png, raster_std_scan_1_07*.png, 
// #figure(image("_artifacts/study.1-rhythmicity-index/raster_std_scan_1_00dark.png"), caption: [Rasters while scanning over $\sigma_{ee}$ and keeping $\mu_{ee}$
zeroed. Left: No PING rhythm in the E population. Middle: Weak PING rhythm
emerging. Right: Strong PING rhythm with clear bursts in the E population.])


With low $W_(ei)$ and $W_(ie)$, the E-I loop is too weak to rhythmically
sustain itself. 

Increasing $W_(ee)$ does three things:

1. It amplifies coincident E spikes into a tighter E burst.
2. That burst finally drives enough I (even through weak $W_(ei)$) to create a
    delayed inhibitory pulse.
3. When inhibition decays, recurrent E ($W_(ee)$) re-seeds the next burst.

That's the PING loop: E burst $arrow.r$ delayed I burst $arrow.r$ E suppression
$arrow.r$ rebound E burst.


== Population rates vs $mu_(ee)$ 


// Gallery: stacked_rate_e_mean_*.png
// #figure(image("_artifacts/study.1-rhythmicity-index/stacked_rate_e_mean_dark.png"), caption: [E Population rate while scanning over $\mu_{ee}$ and keeping $\sigma_{ee}$
zeroed. Bottom: Non-zero oscillatory behaviour, but no PING rhythm in the E population. Middle: Weak PING rhythm emerging. Right: Strong PING rhythm with clear bursts in the E population.])


In Figure 1.3 we can see at low $W_(ee)$ (bottom) that the E population rate is
noisier and not organised into discreet bands, but as $W_(ee)$ increases, the E
population rate becomes more organised into clear bursts. Its expected that PING
effect is not totally absent at low $W_(ee)$, but the rhythmicity is very weak
and noisy.


== E Spikes vs $mu_(ee)$ 


// Gallery: e_spikes_vs_mean_*.png
// #figure(image("_artifacts/study.1-rhythmicity-index/e_spikes_vs_mean_dark.png"), caption: [The total E spikes plotted as a function of $\mu_{ee}$.])


In Figure 1.4 we can see that the total E spikes increases has an interesting
shape, its meaning is unexplored at the present time.

We *do not* normalise the firing rate for experiments at this time, this is to
be explored.


== Autocorrelations vs $mu_(ee)$ 


// Gallery: stacked_autocorr_mean_*.png
// #figure(image("_artifacts/study.1-rhythmicity-index/stacked_autocorr_mean_dark.png"), caption: [Autocorrelations as a function of increasing $\mu_{ee}$ with $\sigma_{ee}$ zeroed.])


Given the E-population rate trace $r(t)$, we center it vertically by subtracting the mean rate $overline(r)$:
$
  \tilde r(t) = r(t) - overline(r)
$

and compute the normalized autocorrelation:
$
  rho(k) =
  \frac{1}{(N-|k|) "Var"(\tilde r)}
  \sum_(t=0)^(N-|k|-1)\tilde r_t \tilde r_(t+|k|)
$

where $k$ is lag in bins ($tau = k Delta t$), and $rho(0)=1$.

The rhythmicity index used in this post is computed in a positive-lag window
($tauin[tau_(min),tau_(max)]$; defaults 20-150 ms):
$
  "RI"=
  cases(
  rho(tau_("first peak")), & "if a peak passes min height + prominence"\
  
  max_(tauin[tau_(min),tau_(max)])rho(tau), & "otherwise"
  )
$


The window in the plots is the window from which we compute the rhythmicity
index, which is the height of the max peak point in the window, excluding the
zero lag peak.


== Rhythmicity metric vs $mu_(ee)$ 


// Gallery: metrics_vs_mean_*.png
// #figure(image("_artifacts/study.1-rhythmicity-index/metrics_vs_mean_dark.png"), caption: [Rhythmicity metric as a function of increasing $\mu_{ee}$ with $\sigma_{ee}$ zeroed.])


We can clearly see that the rhythmicity metric increases with increasing
$mu_(ee)$, which is consistent with the emergence of a stronger PING rhythm.


== Rhythmicity heatmap


// Gallery: heatmap_*.png
// #figure(image("_artifacts/study.1-rhythmicity-index/heatmap_dark.png"), caption: [Rhythmicity heatmap as a function of increasing $\mu_{ee}$ and $\sigma_{ee}$.])


We run the same rhythmicity metric across the whole $mu_(ee)$ and $sigma_(ee)$
parameter space, and plot it as a heatmap. We can see that the rhythmicity is
highest in the top right corner, where both $mu_(ee)$ and $sigma_(ee)$ are
high, which is consistent with the emergence of a strong PING rhythm in that
region of parameter space.


= Parameters


The specification for this simulation is as follows:

// <pre className="bg-gray-100 p-4 rounded">
  {JSON.stringify(config, null, 2)}
// </pre>
