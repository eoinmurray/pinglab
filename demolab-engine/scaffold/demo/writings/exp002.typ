#import "/demolab-engine/build/lib.typ": numbers-table, provenance-footer, video

#let meta = (
  title: "A passive cartpole in MuJoCo",
  date: "2026-06-05",
  description: "A first MuJoCo demo: releasing a cartpole from a small offset and recording the fall as an mp4.",
  collection: "mujoco",
  status: "revising",
)

#let run = json("/artifacts/data/exp002/numbers.json")

#let body = [
  The cartpole is the simplest physics-engine model that still looks like something: a
  cart slides on a frictionless rail with a thin pole hinged on top. With no controller,
  released from a small offset, the pole should topple under gravity while the cart barely
  moves, since the rail absorbs almost all of the hinge's reaction force. We ran a four-second
  passive simulation and recorded the fall.

  == Methods

  The model is a small MJCF cartpole:

  - one slide joint for the cart along the x-axis
  - one hinge joint for the pole about the y-axis
  - damping coefficients $0.05$ and $0.005$ for the cart and pole
  - timestep $0.005 "s"$, simulation length $4 "s"$
  - initial pole offset $theta_0 = #run.cartpole.config.theta0$ rad (≈ #calc.round(run.cartpole.config.theta0 * 180 / calc.pi, digits: 1)°); everything else starts at rest

  The simulation is stepped and a frame sampled every 1/60 s into the video below.

  == Results

  #video("cartpole.mp4", caption: [Passive cartpole released from a small offset of
    $theta_0 = #run.cartpole.config.theta0$ rad: with no controller the pole topples under
    gravity while the cart stays almost stationary on the rail.])

  The pole crosses 60° from vertical at $t approx #calc.round(run.cartpole.fall_time_s, digits: 2) "s"$, ends the
  run essentially horizontal (final angle $#calc.round(run.cartpole.final_angle_deg, digits: 1) degree$), and the
  cart barely moves, only $approx #calc.round(run.cartpole.max_cart_displacement_m * 1e5, digits: 1) times 10^(-5) "m"$
  of recoil over the full four seconds, because almost all the angular momentum of the falling pole is absorbed by
  the rail's reaction force rather than by translating the cart.

  #numbers-table(run.cartpole, title: "Run parameters")

  #provenance-footer(run.cartpole.config)
]
