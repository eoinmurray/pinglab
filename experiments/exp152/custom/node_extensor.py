"""Extensor ROS node: its own single Elevate motor SNNLab population.

Run as a module inside the local experiment container. Reuse the flexor node
lifecycle with the extensor side and its own persistent neural state.
"""

from experiments.exp152.custom.node_flexor import run

if __name__ == "__main__":
    run("extensor")
