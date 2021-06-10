from setuptools import setup, find_packages

setup(name='gymlob',
      version='0.1',
      description='Gym Environment for Limit Order Book Reinforcement Learning Research',
      author='Mahmoud Mahfouz',
      author_email='m.mahfouz17@ic.ac.uk',
      license="MIT license",
      url='https://github.com/mamahfouz/gymlob',
      packages=find_packages(exclude=['contrib', 'docs', 'tests*']),
      install_requires=[
          'numpy',
          'scipy',
          'pandas',
          'matplotlib',
          'plotly',
          'scikit-learn',
          'gym',
          'stable-baselines',
          'torch',
          'tensorboardX',
      ],
      test_suite='tests')