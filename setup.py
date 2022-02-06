from setuptools import setup, find_packages
import pathlib
import mrobo_torch
# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

setup(
    name="mrobo_torch",
    version=mrobo_torch.__version__,
    description="mrobo_torch: Implementation of modern robotics Kinematics Engine in PyTorch",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/hristo-vrigazov/modern_robotics_torch",
    author="Hristo Vrigazov",
    author_email="hvrigazov@gmail.com",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Operating System :: OS Independent"
    ],
    packages=find_packages(exclude=("tests",
                                    "benchmark_images.py")),
    include_package_data=True,
    install_requires=["numpy",
                      "torch",
                      "modern_robotics"],
)
