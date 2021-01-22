from setuptools import setup, find_packages


setup(name='SpCL',
      version='1.0.0',
      description='Self-paced Contrastive Learning with Hybrid Memory for Domain Adaptive Object Re-ID',
      author='Yixiao Ge',
      author_email='geyixiao831@gmail.com',
      url='https://github.com/yxgeee/SpCL',
      install_requires=[
          'numpy', 'torch', 'torchvision',
          'six', 'h5py', 'Pillow', 'scipy',
          'scikit-learn', 'metric-learn', 'faiss_gpu==1.6.3'],
      packages=find_packages(),
      keywords=[
          'Unsupervised Learning',
          'Unsupervised Domain Adaptation',
          'Contrastive Learning',
          'Object Re-identification'
      ])
