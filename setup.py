from setuptools import setup, find_packages

setup(
    name='wormimtools',
    version='0.3.2',
    packages=find_packages(),
    install_requires=["numpy", "scipy", "matplotlib", "pandas", "scikit-image"],
    author='Andrew Moore',
    author_email='moore.andrew0598@gmail.com',
    description='A short description of my package',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/moore-andrew05/imtools',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)