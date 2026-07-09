#import "/demolab-engine/build/lib.typ": numbers-table, provenance-footer, video, data-file

#let meta = (
  title: "A chaotic double pendulum in MuJoCo",
  date: "2026-06-05",
  description: "Two identical double pendulums released side-by-side with initial angles differing by 1e-3 rad: they diverge.",
  collection: "mujoco",
  status: "draft",
)

#let run = json(data-file("exp003/numbers.json"))

#let body = [
  The double pendulum is the textbook example of a deterministic system that is, in
  practice, unpredictable: two arms hinged in series have four state variables
  ($theta_1, theta_2, dot(theta)_1, dot(theta)_2$) and a Lagrangian with one nonlinear
  coupling term, and that is enough for trajectories that start arbitrarily close together
  to separate exponentially. We ran two identical pendulums side by side, started
  $10^(-3)$ rad apart, and measured when and how far they diverge.

  == Methods

  The demo runs two double pendulums at once, side by side in the same MuJoCo scene, with
  identical mass, length, and damping. The only difference is the initial angle of the
  upper arm,

  $ theta_1^A (0) = 2.0, quad theta_1^B (0) = 2.0 + 10^(-3), $

  in radians. The MJCF model is two double-pendulum chains:

  - two chains of two hinge arms each, anchored at $x = plus.minus 0.6 "m"$
  - arms are 0.4 m capsules; a small site marks each tip so its world position can be read
  - timestep 0.002 s with the RK4 integrator, important in the chaotic regime, where forward Euler would let small numerical errors swamp the actual divergence
  - damping 0, so total energy is conserved (give or take RK4's drift)

  Tip separation is measured in each pendulum's _anchor-local_ frame (each tip position
  minus its own anchor) so the constant 1.2 m horizontal offset between the two scenes
  doesn't contaminate the metric.

  == Results

  For the first two seconds or so the two pendulums trace what looks like the same motion.
  Around $t approx #calc.round(run.double_pendulum.separation_time_s, digits: 1) "s"$ the tip-to-tip distance crosses
  the $#run.double_pendulum.config.separation_threshold "m"$ threshold, and from then on they are visibly doing
  completely different things. The Lyapunov-like blow-up is the whole story.

  #video("double_pendulum.mp4", caption: [Two near-identical double pendulums (started
    $10^(-3)$ rad apart) in one MuJoCo scene: they track each other for the first couple of
    seconds, then diverge visibly once the tip separation passes the
    $#run.double_pendulum.config.separation_threshold "m"$ threshold.])

  The maximum separation hits $#calc.round(run.double_pendulum.max_separation_m, digits: 2) "m"$, basically the full
  extent of a single pendulum ($0.8 "m"$), meaning that at some point one tip is at the top of its swing while the
  other's is at the bottom. The final separation of $#calc.round(run.double_pendulum.final_separation_m, digits: 2) "m"$
  is just where they happen to be at $t = #run.double_pendulum.config.duration "s"$; with a chaotic system, "where they
  end up" is meaningless past the divergence time.

  #numbers-table(run.double_pendulum, title: "Run parameters")

  #provenance-footer(run.double_pendulum.config)
]
