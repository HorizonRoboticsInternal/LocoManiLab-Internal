# read the contents of your README file
from os import path

from setuptools import find_packages, setup

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, "README.md"), encoding="utf-8") as f:
    lines = f.readlines()

# remove images from README
lines = [x for x in lines if ".png" not in x]
long_description = "".join(lines)

setup(
    name="robocasa",
    packages=[
        package
        for package in find_packages()
        # g1_bridge (the host-agnostic teleop + recording layer) is no longer
        # shipped from here -- it moved into the g1_backend distribution, which
        # this repo already installs for the sim2real runtime. What stays is
        # robocasa.g1_host: the robosuite/MuJoCo half of the G1 bridge.
        if package.startswith("robocasa")
    ],
    install_requires=[
        "numpy==2.2.5",
        "numba==0.61.2",
        "scipy==1.15.3",
        "mujoco==3.3.1",
        "pygame",
        "Pillow",
        "opencv-python",
        "pyyaml",
        "pynput",
        "tqdm",
        "termcolor",
        "imageio",
        "h5py",
        "lxml",
        "hidapi",
        "lerobot==0.3.3",
        "gymnasium",
        # Required by g1_bridge's vendored WHOLE_BODY_G1 (SONIC) stack
        # (g1_bridge/vendored/wbt/), loaded when that controller is constructed.
        "onnxruntime",
        "loguru",
        # Required by robocasa/scripts/download_g1_checkpoints.py, which fetches
        # the G1 policy checkpoints from the Hub into the g1_backend repo's
        # checkpoints/. Currently satisfied only transitively (via lerobot), and
        # a documented setup step must not rely on a transitive dependency.
        "huggingface_hub>=0.30",
    ],
    eager_resources=["*"],
    include_package_data=True,
    python_requires=">=3",
    description="G1 humanoid benchmark built on RoboCasa365 and robosuite",
    author="Soroush Nasiriany, Sepehr Nasiriany, Abhiram Maddukuri, Yuke Zhu",
    url="https://github.com/LocoManiLab/humanoid_benchmark",
    author_email="abhicm@utexas.edu",
    version="1.0.1",
    long_description=long_description,
    long_description_content_type="text/markdown",
)
