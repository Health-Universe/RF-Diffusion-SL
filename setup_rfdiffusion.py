import os
import sys
import time
import subprocess

def setup_rf_diffusion():
    # Download and set up the necessary parameters and models
    if not os.path.isdir("params"):
        os.makedirs("params", exist_ok=True)
        # Download parameters using aria2c (assuming aria2c is installed in the system)
        subprocess.run([
            "aria2c", "-q", "-x", "16", "http://files.ipd.uw.edu/pub/RFdiffusion/6f5902ac237024bdd0c176cb93063dc4/Base_ckpt.pt",
            "http://files.ipd.uw.edu/pub/RFdiffusion/e29311f6f1bf1af907f9ef9f44b8328b/Complex_base_ckpt.pt",
            "https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar"
        ])
        subprocess.run(["tar", "-xf", "alphafold_params_2022-12-06.tar", "-C", "params"])

    # Clone and install the RFdiffusion repository
    if not os.path.isdir("RFdiffusion"):
        subprocess.run(["git", "clone", "https://github.com/sokrypton/RFdiffusion.git"])
        subprocess.run(["pip", "-q", "install", "jedi", "omegaconf", "hydra-core", "icecream"])
        subprocess.run(["pip", "-q", "install", "dgl", "-f", "https://data.dgl.ai/wheels/cu117/repo.html"])
        subprocess.run(["pip", "-q", "install", "--no-cache-dir", "-r", "RFdiffusion/env/SE3Transformer/requirements.txt"])
        subprocess.run(["pip", "-q", "install", "."], cwd="RFdiffusion/env/SE3Transformer")

    # Install the ColabDesign repository
    if not os.path.isdir("colabdesign"):
        subprocess.run(["pip", "-q", "install", "git+https://github.com/sokrypton/ColabDesign.git@v1.1.1"])

    # Download RFdiffusion parameters
    if not os.path.isdir("RFdiffusion/models"):
        os.makedirs("RFdiffusion/models", exist_ok=True)
        models = ["Base_ckpt.pt", "Complex_base_ckpt.pt"]
        for m in models:
            os.rename(m, os.path.join("RFdiffusion/models", m))

    # Update sys.path and set DGLBACKEND environment variable
    if 'RFdiffusion' not in sys.path:
        os.environ["DGLBACKEND"] = "pytorch"
        sys.path.append('RFdiffusion')

# In the Streamlit app, you can call this function to set up RFdiffusion
# setup_rf_diffusion()
