from __future__ import annotations

import shlex
import subprocess
import modal

app = modal.App("pinglab-streamlit")

streamlit_script_remote_path = "/root/streamlit_app/app.py"

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "streamlit>=1.35.0",
        "matplotlib>=3.10.0",
        "numpy>=2.2.6",
        "pydantic>=2.12.4",
        "scipy>=1.15.3",
        "numba>=0.61.0",
        "pyyaml>=6.0.2",
        "tqdm>=4.67.1",
        "joblib>=1.5.2",
        "tqdm-joblib>=0.0.5",
    )
    .add_local_python_source("pinglab")
    .add_local_dir("src/streamlit_app", remote_path="/root/streamlit_app")
)

@app.function(image=image, max_containers=1)
@modal.concurrent(max_inputs=100)
@modal.web_server(8000)
def serve() -> None:
    target = shlex.quote(streamlit_script_remote_path)
    cmd = (
        "streamlit run "
        f"{target} "
        "--server.port 8000 "
        "--server.address 0.0.0.0 "
        "--server.headless true "
        "--server.enableCORS false "
        "--server.enableXsrfProtection false"
    )
    subprocess.Popen(cmd, shell=True)
