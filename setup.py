from setuptools import setup, find_packages

setup(
    name='blockanalyst',
    version='0.1.0',
    packages=find_packages(),  # Automatically discover and include all packages
    install_requires=[
        # List your dependencies here
    ],
    entry_points={
        'console_scripts': [
            # Define any command-line scripts here
        ],
    },
    author='Mengbi Ye',
    author_email='yemengbi@outlook.com',
    description='A package for block-level urban analysis.',
    long_description='A package for generating urban block geometry and calculate block-level morphological metrics for urban studies.',
    long_description_content_type='text/markdown',
    url='https://github.com/yemengbi/blockanalyst',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        # Add more classifiers as needed
    ],
)