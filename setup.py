from setuptools import setup, find_packages

setup(
    name="cartoonify",
    version="2.0.0",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "cartoonify": ["web/*"],
    },
    entry_points={
        "console_scripts": [
            "cartoonify=cartoonify.cli:main",
        ],
    },
)
