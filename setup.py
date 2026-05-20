from setuptools import setup, find_packages
import os

# Check if we are inside a GitHub Codespace
is_codespace = os.environ.get("CODESPACES") == "true"

# Select the appropriate OpenCV package
opencv_pkg = "opencv-python-headless" if is_codespace else "opencv-python"


setup(
    name="wiresens_backend",
    version="0.1.0",
    author="WiReSens Team",
    description="A backend for wireless tactile sensor data acquisition.",
    long_description="Long description of the WiReSens backend package.",
    long_description_content_type="text/markdown",
    
    # Find the package in the 'src' directory
    package_dir={"": "src"},
    packages=find_packages(where="src"),

    
    # List all the dependencies your package needs to run
    install_requires=[
        "numpy",
        "matplotlib",
        "h5py",
        "json5",
        "pyserial",
        "pyserial-asyncio",
        "bleak",
        "aioconsole",
        opencv_pkg,
        "qrcode[pil]",
        "ipykernel",
        "jupyter"
    ],
    python_requires='>=3.8',
)

