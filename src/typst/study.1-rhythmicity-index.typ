#set document(
  title: "study.1-rhythmicity-index",
  date: datetime(year: 2026, month: 2, day: 17),
)
#metadata((
  title: "study.1-rhythmicity-index",
  date: "2026-02-17",
  description: "Scanning EE weight parameters to map rhythmicity across a PING network.",
)) <meta>

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





The network is setup as follows:

1. 400 E neurons.
2. 100 I neurons.
3. Noisy tonic input to E population only.
4. Weak EI coupling: $W_("ei")$ = $N$(#config.edges.at(2).w.mean, #config.edges.at(2).w.std).
5. Weak IE coupling: $W_("ie")$ = $N$(#config.edges.at(3).w.mean, #config.edges.at(3).w.std).
6. Variable EE coupling: $W_("ee")$ = $N$($mu$, $sigma$) where $mu$ and $sigma$ are scanned over a range of values.

We run two scans:

1. Scan over mean $W_("ee")$ with fixed std $W_("ee")$ of 0.
2. Scan over std $W_("ee")$ with fixed mean $W_("ee")$ of 0.


= Results



== Rasters


#figure(
  grid(
    columns: 3,
    gutter: 4pt,
    image("_artifacts/study.1-rhythmicity-index/raster_mean_scan_2_00_e_to_e_w_mean_0.000000_light.png", width: 60%),
    image("_artifacts/study.1-rhythmicity-index/raster_mean_scan_2_02_e_to_e_w_mean_0.000889_light.png", width: 60%),
    image("_artifacts/study.1-rhythmicity-index/raster_mean_scan_2_06_e_to_e_w_mean_0.002667_light.png", width: 60%),
  ),
  caption: [Rasters scanning over $mu_("ee")$ with $sigma_("ee")$ zeroed. Left: No PING rhythm. Middle: Weak PING rhythm emerging. Right: Strong PING rhythm with clear bursts.],
)


#figure(
  grid(
    columns: 3,
    gutter: 4pt,
    image("_artifacts/study.1-rhythmicity-index/raster_std_scan_1_00_e_to_e_w_std_0.000000_light.png", width: 60%),
    image("_artifacts/study.1-rhythmicity-index/raster_std_scan_1_03_e_to_e_w_std_0.002667_light.png", width: 60%),
    image("_artifacts/study.1-rhythmicity-index/raster_std_scan_1_07_e_to_e_w_std_0.006222_light.png", width: 60%),
  ),
  caption: [Rasters scanning over $sigma_("ee")$ with $mu_("ee")$ zeroed. Left: No PING rhythm. Middle: Weak PING rhythm emerging. Right: Strong PING rhythm with clear bursts.],
)


With low $W_("ei")$ and $W_("ie")$, the E-I loop is too weak to rhythmically
sustain itself. 

Increasing $W_("ee")$ does three things:

1. It amplifies coincident E spikes into a tighter E burst.
2. That burst finally drives enough I (even through weak $W_("ei")$) to create a
    delayed inhibitory pulse.
3. When inhibition decays, recurrent E ($W_("ee")$) re-seeds the next burst.

That's the PING loop: E burst $arrow.r$ delayed I burst $arrow.r$ E suppression
$arrow.r$ rebound E burst.


== Population rates vs $mu_("ee")$ 


#figure(
  image("_artifacts/study.1-rhythmicity-index/stacked_rate_e_mean_scan_2_e_to_e_w_mean_light.png", width: 60%),
  caption: [E Population rate scanning over $mu_("ee")$ with $sigma_("ee")$ zeroed. Bottom: Noisy, no PING rhythm. Middle: Weak PING rhythm emerging. Top: Strong PING rhythm with clear bursts.],
)


In Figure 1.3 we can see at low $W_("ee")$ (bottom) that the E population rate is
noisier and not organised into discreet bands, but as $W_("ee")$ increases, the E
population rate becomes more organised into clear bursts. Its expected that PING
effect is not totally absent at low $W_("ee")$, but the rhythmicity is very weak
and noisy.


== E Spikes vs $mu_("ee")$ 


#figure(
  image("_artifacts/study.1-rhythmicity-index/e_spikes_vs_mean_scan_2_e_to_e_w_mean_light.png", width: 60%),
  caption: [The total E spikes plotted as a function of $mu_("ee")$.],
)



In Figure 1.4 we can see that the total E spikes increases has an interesting
shape, its meaning is unexplored at the present time.

We *do not* normalise the firing rate for experiments at this time, this is to
be explored.


== Autocorrelations vs $mu_("ee")$ 


#figure(
  image("_artifacts/study.1-rhythmicity-index/stacked_autocorr_mean_scan_2_e_to_e_w_mean_light.png", width: 60%),
  caption: [Autocorrelations as a function of increasing $mu_("ee")$ with $sigma_("ee")$ zeroed.],
)



Given the E-population rate trace $r(t)$, we center it vertically by subtracting the mean rate $overline(r)$:
$
  tilde(r)(t) = r(t) - overline(r)
$

and compute the normalized autocorrelation:
$
  rho(k) =
  frac(1, (N-|k|) "Var"(tilde(r)))
  sum_(t=0)^(N-|k|-1) tilde(r)_t tilde(r)_(t+|k|)
$

where $k$ is lag in bins ($tau = k Delta t$), and $rho(0)=1$.

The rhythmicity index used in this post is computed in a positive-lag window
($tau in [tau_("min"), tau_("max")]$; defaults 20-150 ms):
$
  "RI"=
  cases(
  rho(tau_("first peak")), & "if a peak passes min height + prominence"\
  
  max_(tau in [tau_("min"), tau_("max")]) rho(tau), & "otherwise"
  )
$


The window in the plots is the window from which we compute the rhythmicity
index, which is the height of the max peak point in the window, excluding the
zero lag peak.


== Rhythmicity metric vs $mu_("ee")$ 


#figure(
  image("_artifacts/study.1-rhythmicity-index/metrics_vs_mean_scan_2_e_to_e_w_mean_light.png", width: 60%),
  caption: [Rhythmicity metric as a function of increasing $mu_("ee")$ with $sigma_("ee")$ zeroed.],
)



We can clearly see that the rhythmicity metric increases with increasing
$mu_("ee")$, which is consistent with the emergence of a stronger PING rhythm.


== Rhythmicity heatmap


#figure(
  image("_artifacts/study.1-rhythmicity-index/heatmap_ee_mean_std_autocorr_peak_light.png", width: 60%),
  caption: [Rhythmicity heatmap as a function of increasing $mu_("ee")$ and $sigma_("ee")$.],
)



We run the same rhythmicity metric across the whole $mu_("ee")$ and $sigma_("ee")$
parameter space, and plot it as a heatmap. We can see that the rhythmicity is
highest in the top right corner, where both $mu_("ee")$ and $sigma_("ee")$ are
high, which is consistent with the emergence of a strong PING rhythm in that
region of parameter space.


= Parameters


The specification for this simulation is as follows:

// <pre className="bg-gray-100 p-4 rounded">
  {JSON.stringify(config, null, 2)}
// </pre>
