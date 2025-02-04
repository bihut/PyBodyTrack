from setuptools import setup, find_packages

setup(
    name="pyBodyTrack",
    version="2025.2.1",
    author="Angel Ruiz Zafra",
    author_email="angelrzafra@gmail.com",
    description="A Python package for multi-algorithm motion quantification and tracking in videos",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/bihut/pyBodyTrack",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "opencv-python",
        "scipy",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)
#generar los archivos de distribución
#python setup.py sdist bdist_wheel
#instalar twine
#pip install twine
#subir el paquete a pypi
#twine upload dist/*