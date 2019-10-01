from setuptools import setup

setup(name='splinefit',
      version='1.0b',
      description='Spline Fit.',
      url='http://github.com/ooreilly/splinefit',
      author='Ossian O\'Reilly',
      author_email='ooreilly@usc.edu',
      license='MIT',
      packages=['splinefit'],
      scripts=['splinefit/bin/tsurfmsh', 
               'splinefit/bin/mshvtk', 
               'splinefit/bin/sfbbox',
               'splinefit/bin/sfbnd',
               'splinefit/bin/sfproj',
               'splinefit/bin/sfrot',
               'splinefit/bin/sfseg',
               'splinefit/bin/sffbnd',
               'splinefit/bin/sffsrf',
               'splinefit/bin/sfiges',
               'splinefit/bin/sfgeo',
               'splinefit/bin/sfbuild',
               ],
    install_requires=[
    'numpy',
    'scipy==1.1',
    'matplotlib'
    ],
      zip_safe=False)
