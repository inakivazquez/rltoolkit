from setuptools import setup, find_packages

setup(
    name='rltoolkit',
    version='1.0.0',
    description='A toolkit for reinforcement learning',
    author='Inaki Vazquez',
    author_email='ivazquez@deusto.es',
    packages=find_packages(),
    install_requires=[
        'torch',
        'gymnasium',
        'stable-baselines3'
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: Python :: 3.8',
    ],
    entry_points={
        'console_scripts': [
            'check-cuda=rltoolkit.check_cuda:main',
            'test-gymnasium=rltoolkit.test_gymnasium:main',
            'test-sb3=rltoolkit.test_sb3:main',
            'train-sb3=rltoolkit.train_sb3:main',
            'eval-sb3=rltoolkit.eval_sb3:main',
        ],
    },
)