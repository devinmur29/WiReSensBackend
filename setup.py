from setuptools import setup, find_packages


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
        "h5py",
        "json5",
        "pyserial",
        "pyserial-asyncio",
        "bleak",
        "aioconsole",
        "opencv-python",
        "qrcode[pil]",
    ],
    python_requires='>=3.8',
)

