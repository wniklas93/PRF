#!/usr/bin/env python

import setuptools

if __name__ == "__main__":
    setuptools.setup(
        name='filterbank',
        version='0.3a1',

        description='Perfect Reconstruction Filter Banks',
        author='Niklas Winter',
        author_email='niklas.winter@audiolabs-erlangen.de',

        license='proprietary',
        packages=setuptools.find_packages(),

        install_requires=[
            'numpy>=1.21.5',
            'scipy>=1.8.0',
            'scikit-image>=0.19.1',
            'matplotlib >= 3.5.1'
        ],

        extras_require={
            'tests': [
                'pytest>=6.2.5',
                'pytest-cov>=3.0.0',
                'pytest-pycodestyle>=2.2.0',
                'tox>=3.24.4',
                'numpy>=1.19.5',
            ]
        },

        classifiers=[
            'Development Status :: 4 - Beta',
            'Intended Audience :: Telecommunications Industry',
            'Intended Audience :: Science/Research',
            'Programming Language :: Python :: 3.8',
            'Topic :: Multimedia :: Sound/Audio :: Analysis',
            'Topic :: Multimedia :: Sound/Audio :: Sound Synthesis'
        ],

        zip_safe=True,
        include_package_data=True,
    )