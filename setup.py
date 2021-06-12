import os
import sys
import platform
from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext as _build_ext


def setup_package():
    compiler_directives = {
        "language_level": 3,
        "embedsignature": True,
    }

    annotate = False

    class new_build_ext(_build_ext):
        def finalize_options(self):
            # Defer the import of Cython and NumPy until after setup_requires
            from Cython.Build.Dependencies import cythonize
            import numpy

            self.distribution.ext_modules[:] = cythonize(
                self.distribution.ext_modules,
                compiler_directives=compiler_directives,
                annotate=annotate,
            )
            if not self.include_dirs:
                self.include_dirs = []
            elif isinstance(self.include_dirs, str):
                self.include_dirs = [self.include_dirs]
            self.include_dirs.append(numpy.get_include())
            super().finalize_options()

    metadata = dict(
        name="pyene",
        version='0.1',
        description='Python Energy and Networks Engine - pyene.',
        url='git@gitlab.hydra.org.uk:futuredams/test-case/DAMSEnergy.git',
        author='Dr. Eduardo Alejandro Martínez Ceseña, \
            Dr. Jose Nicolas Melchor Gutierrez',
        author_email='Eduardo.MartinezCesena@manchester.ac.uk, \
            jose.melchorgutierrez@manchester.ac.uk',
        packages=find_packages(),
        package_data={'pyene': ['json/*.json']},
        install_requires=['click', 'pandas', 'pyomo', 'pypsa'],
        extras_require={"test": ["pytest"]},
        cmdclass={"build_ext": new_build_ext},
        # use_scm_version=True,
        entry_points='''
        [console_scripts]
        pyene=pyene.cli:cli
        ''',
    )
    
    config = parse_optional_arguments()

    annotate = config["annotate"]

    if config["profile"]:
        compiler_directives["profile"] = True

    if config["trace"]:
        compiler_directives["linetrace"] = True
    
    metadata["ext_modules"] = ext_modules = []

    if config["glpk"]:
        ext_modules.append(Extension("pyene.engines.cython._glpk", ["pyene/engines/_glpk.pyx"],
        include_dirs=[findglpkheaderpath()],
        library_dirs=[findglpklibrarypath()], 
        libraries=["glpk"],))

    if config["clp"]:
        if platform.system() == "Windows":
            ext_modules.append(Extension("pyene.engines.cython.cpp_energy_wrapper", ["pyene/engines/cpp_energy_wrapper.pyx"],
            include_dirs=[os.path.dirname(os.path.abspath(__file__))+'\pyene\engines\external files\\armadillo-10.1.2\include',
                      os.path.dirname(os.path.abspath(__file__))+'\pyene\engines\external files\\boost_1_74_0',
                      os.path.dirname(os.path.abspath(__file__))+"\pyene\engines\external files\Clp\include",
                      os.path.dirname(os.path.abspath(__file__))+"\pyene\engines\external files\CoinUtils\include",
                      os.path.dirname(os.path.abspath(__file__))+"\pyene\engines\external files\BuildTools\headers"],
            libraries=['libClp', 'libCoinUtils', 'libopenblas'],
            library_dirs=[os.path.dirname(os.path.abspath(__file__))+"\pyene\engines\external files\Clp\lib",  os.path.dirname(os.path.abspath(__file__))+"\pyene\engines\external files\\armadillo-10.1.2\lib_win64"],
            define_macros= [('ARMA_DONT_USE_WRAPPER', None),
                            ('ARMA_USE_LAPACK', None),
                            ('ARMA_USE_BLAS', None)]))
        elif platform.system() == "Linux":
            ext_modules.append(Extension("pyene.engines.cython.cpp_energy_wrapper", ["pyene/engines/cpp_energy_wrapper.pyx"],
            include_dirs=["pyene/engines/external files/boost_1_74_0",
                      "pyene/engines/external files/Clp/include",
                      "pyene/engines/external files/CoinUtils/include",
                      "pyene/engines/external files/BuildTools/headers"],
            libraries=['Clp', 'armadillo'],
            ))
    setup(**metadata)

def parse_optional_arguments():
    config = {
        "glpk": True,
        "clp": True,
        "annotate": False,
        "profile": False,
        "trace": False,
    }

    if "--without-glpk" in sys.argv:
        config["glpk"] = False
        sys.argv.remove("--without-glpk")
    
    if "--without-clp" in sys.argv:
        config["clp"] = False
        sys.argv.remove("--without-clp")

    if "--annotate" in sys.argv:
        config["annotate"] = True
        sys.argv.remove("--annotate")

    if "--enable-profiling" in sys.argv:
        config["profile"] = True
        sys.argv.remove("--enable-profiling")

    if "--enable-trace" in sys.argv:
        config["trace"] = True
        sys.argv.remove("--enable-trace")
    return config

def findglpkheaderpath():
    inc_arg = "-I"
    for arg in sys.argv:
        if arg.startswith(inc_arg) and len(arg) > len(inc_arg):
            return arg[len(inc_arg):]

    # Finding glpk header path
    pythonpath = os.path.split(sys.executable)[0]
    if len(pythonpath.rsplit('/b', 1)) > 1:
        aux1 = pythonpath.rsplit('/b', 1)[1]
        aux2 = pythonpath.rsplit('/b', 1)[0]
        print(aux1)
        print(aux2)
        if aux1 == 'in':
            pythonpath = aux2
    trypaths = [pythonpath+'/Library/include/glpk.h',
                pythonpath+'/include/glpk.h',
                pythonpath+'/Library/include/glpk.h',
                pythonpath+'/include/glpk.h']
    glpkpath = None
    for paths in trypaths:
        if os.path.isfile(paths):
            glpkpath = paths[:-7]
            break
    if glpkpath is None:
        print('Path for GLPK header has not been found in the predefined \
            directories')
    
    return glpkpath

def findglpklibrarypath():   
    lib_arg = "-L"
    for arg in sys.argv:
        if arg.startswith(lib_arg) and len(arg) > len(lib_arg):
            return arg[len(lib_arg):]

    # Finding glpk header path
    pythonpath = os.path.split(sys.executable)[0]
    if len(pythonpath.rsplit('/b', 1)) > 1:
        aux1 = pythonpath.rsplit('/b', 1)[1]
        aux2 = pythonpath.rsplit('/b', 1)[0]
        print(aux1)
        print(aux2)
        if aux1 == 'in':
            pythonpath = aux2
    trypaths = [pythonpath+'/Library/lib/glpk.lib',
                pythonpath+'/libs/glpk.lib',
                pythonpath+'/Library/lib/libglpk.so',
                pythonpath+'/lib/libglpk.so']
    glpkpath = None
    for paths in trypaths:
        if os.path.isfile(paths):
            glpkpath = paths[:-9]
            break
    if glpkpath is None:
        print('Path for GLPK library has not been found in the predefined \
            directories')
    
    return glpkpath

if __name__ == "__main__":
    setup_package()
