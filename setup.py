from setuptools import setup, find_packages

setup(
    name='track_transportation',
    version='0.0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'opencv-python',
        'torch',
        'matplotlib',
        'ultralytics',
        'cvzone',
        'tqdm',
    ],
)
