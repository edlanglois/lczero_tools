"""Package setup script."""
import setuptools

setuptools.setup(
    name='lczero-tools',
    version='0.1',
    packages=setuptools.find_packages('src'),
    package_dir={'': 'src'},
    install_requires=[
        'dask',
        'numpy',
        'python-chess',
    ],
    extras_require={
        'tf': ['tensorflow'],
        'tf-gpu': ['tensorflow-gpu'],
        'torch': ['torch'],
    },
    setup_requires=[],
    tests_require=[],
)
