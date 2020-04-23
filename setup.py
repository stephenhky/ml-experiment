
from setuptools import setup

def readme():
    with open('README.md') as f:
        return f.read()


def install_requirements():
    return [package_string.strip() for package_string in open('requirements.txt', 'r')]


def package_description():
    text = open('README.md', 'r').read()
    startpos = text.find('## Introduction')
    return text[startpos:]


setup(name='ml-experiment',
      version="0.0.1a01",
      description="Machine Learning Experiment Framework",
      long_description="Fast prototyping of machine learning experiments",
      classifiers=[
          "Topic :: Scientific/Engineering :: Artificial Intelligence",
          "Topic :: Scientific/Engineering :: Mathematics",
          "Topic :: Software Development :: Libraries :: Python Modules",
          "Programming Language :: Python :: 3.5",
          "Programming Language :: Python :: 3.6",
          "Programming Language :: Python :: 3.7",
          "Programming Language :: Python :: 3.8",
          "License :: OSI Approved :: MIT License",
      ],
      keywords="machine learning",
      url="https://github.com/stephenhky/ml-experiment",
      author="Kwan-Yuet Ho",
      author_email="stephenhky@yahoo.com.hk",
      license='MIT',
      packages=['mlexpt',
                'mlexpt.utils',
                'mlexpt.data',
                'mlexpt.metrics',
                'mlexpt.ml',
                'mlexpt.ml.classifiers',
                'mlexpt.ml.encoders'],
      install_requires=install_requirements(),
      tests_require=[
          'unittest2',
      ],
      scripts=['script/CSV2JSON',
               'script/RunExperiments'],
      include_package_data=True,
      test_suite="test",
      zip_safe=False)
