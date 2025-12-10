
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import shutil
import sys
from pinglab.plots.raster import save_raster
from pinglab.run.run_network import run_network
from pinglab.types import NetworkResult
import yaml
from sklearn.linear_model import Ridge

from pinglab.plots.styles import save_both, figsize

# Add the experiment directory to path for local imports
sys.path.insert(0, str(Path(__file__).parent))
from local.model import LocalConfig
from local.generate_image import generate_image
from local.image_to_external_input import image_to_external_input
from local.reconstruct_image_from_spikes import reconstruct_image_from_spikes
from local.build_feedforward_current import build_feedforward_current
from local.collect_dataset import collect_dataset

def main() -> None:
    root = Path(__file__).parent
    data_path = root / "data"
    if data_path.exists():
        shutil.rmtree(data_path)
    data_path.mkdir(parents=True, exist_ok=True)
    config_path = root / "config.yaml"
    with config_path.open() as f:
        data = yaml.safe_load(f)
    config = LocalConfig.model_validate(data)

    image = generate_image()

    def plot_input_image():
      _, ax = plt.subplots(1, 1, figsize=figsize)
      ax.imshow(image, cmap='gray')
      plt.tight_layout()

    save_both(data_path / "input_image", plot_input_image)

    external_input_A = image_to_external_input(image, config)

    result_A: NetworkResult = run_network(config.base, external_input=external_input_A)

    save_raster(
        result_A.spikes,
        path=data_path / "raster_A.png",
        external_input=external_input_A,
        dt=config.base.dt,
    )

    rate_image_A = reconstruct_image_from_spikes(
        result_A.spikes,
        N_E=config.base.N_E,
        T_window=(500.0, 1000.0),   # avoid initial transients
    )

    def plot_reconstruction():
        fig, axes = plt.subplots(1, 2, figsize=(figsize[0], figsize[1] / 2))
        axes[0].set_title("Input")
        axes[0].imshow(image, cmap="gray")
        axes[0].axis("off")

        axes[1].set_title("Rate reconstruction")
        axes[1].imshow(rate_image_A, cmap="gray")
        axes[1].axis("off")

        plt.tight_layout()

    save_both(data_path / "reconstruction_A_only", plot_reconstruction)

    ff_current = build_feedforward_current(
        result_A.spikes,
        N_E=config.base.N_E,
        N_I=config.base.N_I,
        T=config.base.T,
        dt=config.base.dt,
        delay_ms=5.0,
        pulse_ms=3.0,
        w_ff=0.3,
    )

    # 3) Baseline tonic input for B (no image; just drive it into gamma)
    num_steps = int(config.base.T / config.base.dt)
    N = config.base.N_E + config.base.N_I
    external_input_B = np.zeros((num_steps, N), dtype=np.float32)

    baseline_I_E = 0.7   # lower than your usual 1.2-ish
    baseline_I_I = 0.5

    # simple: same constant I_E / I_I for all neurons
    external_input_B[:, :config.base.N_E] = baseline_I_E
    external_input_B[:, config.base.N_E:] = baseline_I_I

    # add noise if you like
    if config.default_inputs.noise > 0:
        external_input_B += np.random.normal(
            loc=0.0,
            scale=config.default_inputs.noise,
            size=(num_steps, N),
        ).astype(np.float32)

    # external_input_B[:] = 0.0

    # add the feedforward current from A
    external_input_B += ff_current

    # 4) Run Network B
    result_B: NetworkResult = run_network(config.base, external_input=external_input_B)

    save_raster(result_B.spikes, path=data_path / "raster_B.png")

    rate_image_B = reconstruct_image_from_spikes(
        result_B.spikes,
        N_E=config.base.N_E,
        T_window=(500.0, 1000.0),
    )

    def plot_recon_B():
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].set_title("Input image")
        axes[0].imshow(image, cmap="gray"); axes[0].axis("off")

        axes[1].set_title("A (direct rate recon)")
        axes[1].imshow(rate_image_A, cmap="gray"); axes[1].axis("off")

        axes[2].set_title("B (after feedforward)")
        axes[2].imshow(rate_image_B, cmap="gray"); axes[2].axis("off")

        plt.tight_layout()

    save_both(data_path / "reconstruction_AB", plot_recon_B)

    # X_train, Y_train = collect_dataset(config, num_images=50, phase_ms=0.0)
    # decoder = Ridge(alpha=1.0)
    # decoder.fit(X_train, Y_train)

    # X_test, Y_test = collect_dataset(config, num_images=1, phase_ms=0.0)
    # Y_hat = decoder.predict(X_test)[0]

    # side = int(np.sqrt(config.base.N_E))
    # target = Y_test[0].reshape(side, side)
    # recon  = Y_hat.reshape(side, side)

    # def plot_fn():
    #     fig, axes = plt.subplots(1, 2, figsize=figsize)
    #     axes[0].set_title("Target (original image)")
    #     axes[0].imshow(target, cmap="gray", vmin=0.0, vmax=1.0)
    #     axes[0].axis("off")

    #     axes[1].set_title("Reconstruction from B")
    #     axes[1].imshow(recon, cmap="gray", vmin=0.0, vmax=1.0)
    #     axes[1].axis("off")
    #     plt.tight_layout()

    # save_both(data_path / "reconstruction_phase_0", plot_fn)

    # phases_ms = np.linspace(-15.0, 15.0, 13)   # e.g. -15 .. +15 ms
    # errors = []

    # for phi in phases_ms:
    #     X_phi, Y_phi = collect_dataset(config, num_images=20, phase_ms=phi)
    #     Y_hat_phi = decoder.predict(X_phi)

    #     mse = np.mean((Y_hat_phi - Y_phi) ** 2)
    #     errors.append(mse)

    # errors = np.array(errors)

    # def plot_mse_vs_phase():
    #     plt.figure()
    #     plt.plot(phases_ms, errors, marker="o")
    #     plt.xlabel("Phase offset (ms)")
    #     plt.ylabel("Reconstruction MSE")
    #     plt.tight_layout()
    
    # save_both(data_path / "mse_vs_phase", plot_mse_vs_phase)


if __name__ == "__main__":
    main()
