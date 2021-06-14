LIBRARY_INC=C:\ANACONDA3\Library\include
LIBRARY_LIB=C:\ANACONDA3\Library\lib

run: compile
	python tests/test_core.py

compile:
	'C:\ANACONDA3\python.exe' setup.py build_ext -I'C:\ANACONDA3\Library\include' -L'C:\ANACONDA3\Library\lib' --inplace develop