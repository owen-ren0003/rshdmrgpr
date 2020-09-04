from setuptools import setup

setup(# Needed to silence warnings (and to be a worthwhile package)
    name='rshdmrgpr',
    url='git@github.com/owen-ren0003/rshdmrgpr.git',
    author='Owen Ren',
    author_email='owen.z.ren@1234@gmail.com',
    # Needed to actually package something
    packages=['rshdmrgpr'],
    # Needed for dependencies
    install_requires=['numpy', 'pandas', 'matplotlib', 'scikit-learn'],
    # *strongly* suggested for sharing
    version='0.0.1',
    # The license can be anything you like
    license='MIT',
    description='RS-HDMR-GPR',
    zip_safe=False
    # We will also need a readme eventually (there will be a warning)
    # long_description=open('README.txt').read(),
)
