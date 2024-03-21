import setuptools

with open("README.md","r",encoding="utf-8") as f:
    long_discription=f.read()

__version__ = "0.0.1"

REPO_NAME = "track_transportation"
AUTHOR_USER_NAME = "prthmgoyl"
AUTHOR_EMAIL = "prthmgoyl@gmail.com"
SRC_REPO = "track_transportation"

setuptools.setup(
    name=SRC_REPO,
    version="0.0.1",
    author=AUTHOR_USER_NAME ,
    author_email=AUTHOR_EMAIL,
    description="a small python project",
    long_description=long_discription,
    long_discription_content="text/markdown",
    url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    project_urls={
        "Bug Tracker": f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}/issues"
    },
    package_dir={"":"src"},
    packages=setuptools.find_packages(where="src"),
    install_requires=[
        'numpy',
        'opencv-python',
        'torch',
        'matplotlib',
        'ultralytics',
        'cvzone',
        'tqdm',
    ]
)
