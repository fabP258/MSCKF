from setuptools import setup, find_packages

setup(
    name="msckf",
    version="0.0.1",
    description="A Multi-State Constraint Kalman Filter implementation for visual odometry",
    author="Fabio Petzenhauser",
    author_email="fabio894@gmail.com",
    paxkages=find_packages(),
    install_requires=["numpy"],
    extras_require={"dev": ["pytest"]},
)
