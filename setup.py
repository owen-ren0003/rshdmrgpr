from setuptools import setup

setup(
    name='rshdmrgpr',
    url='git@github.com/owen-ren0003/rshdmrgpr.git',
    author='Owen Ren, Mohamed Boussadi Ali, Sergei Manzhos',
    author_email='owen.z.ren@1234@gmail.com',
    packages=['rshdmrgpr'],
    install_requires=['numpy', 'pandas', 'matplotlib', 'scikit-learn'],
    version='0.0.1',
    license='MIT',  # This needs to be updated to the Journal requirements
    description='RS-HDMR-GPR',
    zip_safe=False,
    long_description=open('readme.md').read(),
)
