# -*- coding: utf-8 -*-
from distutils.core import setup
setup(
    name='multilabel-metrics',
    version='0.0.1',
    scripts=['multilabel-metrics'],
    author=u'Abhishek Verma',
    author_email='abhishek_verma@hotmail.com',
	url='https://github.com/hell-sing/multi-label-metrics',
	long_description=open('README.txt').read(),
	py_modules=['mlmetrics'],
    license='GNU_GPL licence, see LICENCE.txt',
    description='Multilabel classification metrics for Python',
	zip_safe=False
)
