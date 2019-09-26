from setuptools import setup

setup(name='splinefit',
      version='0.1',
      description='Spline Fit.',
      url='http://github.com/ooreilly/splinefit',
      author='Ossian O\'Reilly',
      author_email='ooreilly@usc.edu',
      license='MIT',
      packages=['splinefit'],
      scripts=['splinefit/bin/tsurfmsh', 'splinefit/bin/mshvtk', 
               'splinefit/bin/sfbbox',
               'splinefit/bin/sfbnd',
               'splinefit/bin/sfbnd',
               'splinefit/bin/sfrot',
               'splinefit/bin/sfseg',
               ],
      zip_safe=False)
