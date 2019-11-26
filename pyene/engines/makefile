LIBRARY_INC=C:\ANACONDA3\Library\include
LIBRARY_LIB=C:\ANACONDA3\Library\lib

run: compile
	python tests/test_core.py

compile:
	python setup.py build_ext -IC:\ANACONDA3\Library\include -LC:\ANACONDA3\Library\lib --inplace develop