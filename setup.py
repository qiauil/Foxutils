import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="foxutils",
    version="0.0.26",
    author="Qiauil",
    author_email="qiangliu.7@outlook.com",
    description="Utils for PyTorch based deep-learning study",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://qiauil.github.io/",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
