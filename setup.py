import os
import sys
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
            Dr. Jose NicolasMelchor Gutierrez',
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
    
    if config["glpk"]:
        metadata["ext_modules"] = ext_modules = [
        Extension("pyene.engines._glpk", ["pyene/engines/_glpk.pyx"],
        include_dirs=[findglpkheaderpath()],
        library_dirs=[findglpklibrarypath()], 
        libraries=["glpk"],),
        ]

    setup(**metadata)

def parse_optional_arguments():
    config = {
        "glpk": True,
        "annotate": False,
        "profile": False,
        "trace": False,
    }

    if "--with-glpk" in sys.argv:
        config["glpk"] = True
        sys.argv.remove("--with-glpk")
    elif "--without-glpk" in sys.argv:
        config["glpk"] = False
        sys.argv.remove("--without-glpk")

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
    # Finding glpk header path
    pythonpath = os.path.split(sys.executable)[0]
    trypaths = [pythonpath+'\Library\include\glpk.h',\
                pythonpath+'\include\glpk.h']
    glpkpath = None
    for paths in trypaths:
        if os.path.isfile(paths):
            glpkpath = paths[:-7]
            break
    if glpkpath is None:
        sys.exit('Path for GLPK header has not been found')
    
    return glpkpath

def findglpklibrarypath():   
    # Finding glpk header path
    pythonpath = os.path.split(sys.executable)[0]
    trypaths = [pythonpath+'\Library\lib\glpk.lib',\
                pythonpath+'\libs\glpk.lib']
    glpkpath = None
    for paths in trypaths:
        if os.path.isfile(paths):
            glpkpath = paths[:-9]
            break
    if glpkpath is None:
        sys.exit('Path for GLPK library has not been found')
    
    return glpkpath

if __name__ == "__main__":
    setup_package()
