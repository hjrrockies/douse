"""The setup script."""

from setuptools import setup, find_packages

with open('README.md') as readme_file:
    readme = readme_file.read()

setup(
    author="Hayden Ringer",
    author_email='hringer@vt.edu',
    description="Domains of Unknown Shape from Eigenvalues",
    long_description=readme + '\n\n',# + history,
    include_package_data=True,
    name='douse',
    packages=find_packages(),
    version='0.0.1',
)
