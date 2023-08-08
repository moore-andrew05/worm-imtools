from setuptools import setup, find_packages

setup(
    name='wormimtools',
    version='0.1',
    packages=find_packages(),
    author='Your Name',
    author_email='your.email@example.com',
    description='A short description of my package',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/imaging-tools',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)