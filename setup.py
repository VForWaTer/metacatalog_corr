from setuptools import setup, find_packages
from setuptools.command.develop import develop
from setuptools.command.install import install


def requirements():
    with open('requirements.txt') as f:
        return f.read().split('\n')


def version():
    with open('VERSION') as f:
        return f.read().strip()


def readme():
    with open('README.md') as f:
        return f.read()


def install_and_migrate():
    try:
        from metacatalog_corr import manage
    except ModuleNotFoundError:
        pass

    try:
        manage.install(verbose=True)
    except:
        pass


class PostDevelopCommand(develop):
    def run(self):
        install_and_migrate()
        develop.run(self)


class PostInstallCommand(install):
    def run(self):
        install_and_migrate()
        install.run(self)


setup(
    name='metacatalog_corr',
    author='Mirko Mälicke',
    author_email='mirko.maelicke@kit.edu',
    license='GPL v3',
    install_requires=requirements(),
    version=version(),
    description='Correlation metric extenstion for Metacatalog',
    long_description=readme(),
    long_description_content_type='text/markdown',
    packages=find_packages(),
    cmdclass={
        'develop': PostDevelopCommand,
        'install': PostInstallCommand
    },
    include_package_data=True
)