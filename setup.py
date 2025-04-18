from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='scnet',
    version='0.2.2.2',
    packages=find_packages(),
    include_package_data=True,  # Include data files
    package_data={
        # Include all files within the data directory under the your_package namespace
        'scNET': ['Data/*',"KNNs/*","Embedding/*","Models/*"]
    },
    install_requires=[
        # 'torch==2.2.2', # install manually by pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu118
        'torch-geometric==2.5.3',
        'pandas>=2.2.1',
        'numpy==1.26.4',
        'networkx>=3.1',
        'scanpy>=1.11.0',
        'scikit-learn>=1.4.1',
        'gseapy>=1.1.6',
        'matplotlib>=3.8.0',
        'igraph',
        'leidenalg',
        'tqdm',
        'gdown'
    ],
    author='Ron Sheinin',
    description='Our method employs a unique dual-graph architecture based on graph neural networks (GNNs), enabling the joint representation of gene expression and PPI network data',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/madilabcode/scNET'
)