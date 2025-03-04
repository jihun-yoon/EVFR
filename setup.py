import setuptools

setuptools.setup(
    name="evfr",
    version="0.1.0",
    author="Jihun Yoon",
    author_email="yjh4374@gmail.com",
    description="Exploring Video Frame Redundancy library",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/jihun-yoon/EVFR",
    packages=setuptools.find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "numpy==2.2.3",
        "transformers==4.49.0",
        "Pillow==11.1.0",
        "tqdm==4.67.1",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Video",
    ],
)