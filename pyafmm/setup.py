from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import platform
import pybind11
import numpy

class CustomBuildExt(build_ext):
    """Custom build extension for handling compiler flags"""
    def build_extensions(self):
        compiler_type = self.compiler.compiler_type
        
        for ext in self.extensions:
            if compiler_type == 'msvc':
                # MSVC specific flags
                cpp_flags = [
                    '/O2',          # Optimize for speed
                    '/W3',          # Warning level
                    '/GL',          # Whole program optimization
                    '/EHsc',        # Exception handling model
                    '/std:c++17',   # C++ 17 standard
                ]
                c_flags = [
                    '/O2',          # Optimize for speed
                    '/W3',          # Warning level
                    '/GL',          # Whole program optimization
                ]
                ext.extra_link_args = [
                    '/LTCG',        # Link-time code generation
                ]
                ext.define_macros.append(('WIN32', '1'))
                
                # Set source-specific compiler flags
                for source in ext.sources:
                    if source.endswith('.cpp'):
                        self._set_source_flags(source, cpp_flags)
                    elif source.endswith('.c'):
                        self._set_source_flags(source, c_flags)
                
            else:  # gcc, mingw, clang
                cpp_flags = [
                    '-O3',              # Highest optimization level
                    '-Wall',            # All warnings
                    '-Wextra',          # Extra warnings
                    '-std=c++17',       # C++ 17 standard
                    '-fvisibility=hidden',
                    '-fPIC',           # Position independent code
                ]
                c_flags = [
                    '-O3',             # Highest optimization level
                    '-Wall',           # All warnings
                    '-fPIC',           # Position independent code
                ]
                ext.extra_link_args = [
                    '-lpthread',        # POSIX threading
                    '-lm',             # Math library
                ]
                
                # Set source-specific compiler flags
                for source in ext.sources:
                    if source.endswith('.cpp'):
                        self._set_source_flags(source, cpp_flags)
                    elif source.endswith('.c'):
                        self._set_source_flags(source, c_flags)
                
                # Platform-specific additions
                if platform.system() == "Darwin":  # macOS
                    if source.endswith('.cpp'):
                        self._append_source_flags(source, ['-stdlib=libc++'])
                    ext.extra_link_args.extend(['-stdlib=libc++'])
                    
        build_ext.build_extensions(self)
    
    def _set_source_flags(self, source, flags):
        if not hasattr(self, '_source_flags'):
            self._source_flags = {}
        self._source_flags[source] = flags.copy()
    
    def _append_source_flags(self, source, flags):
        if not hasattr(self, '_source_flags'):
            self._source_flags = {}
        if source not in self._source_flags:
            self._source_flags[source] = []
        self._source_flags[source].extend(flags)
    
    def build_extension(self, ext):
        sources = ext.sources
        if sources is None or not len(sources):
            return
        
        # Apply source-specific flags if any
        if hasattr(self, '_source_flags'):
            for source in sources:
                if source in self._source_flags:
                    extra_args = self._source_flags[source]
                    if self.compiler.compiler_type == 'msvc':
                        if source.endswith('.c'):
                            self.compiler.compile([source], extra_postargs=extra_args)
                        else:
                            self.compiler.compile([source], extra_postargs=extra_args)
                    else:
                        self.compiler.compile([source], extra_postargs=extra_args)
        
        # Continue with regular build
        build_ext.build_extension(self, ext)

# Define source files
sources = [
    'pybind_afmm.cpp',  # Your PyBind11 wrapper (C++)
    'afmm.c',             # Original AFMM implementation (C)
]

# Define include directories
include_dirs = [
    '.',
    pybind11.get_include(),
    pybind11.get_include(user=True),
    numpy.get_include(),
]

# Define macros
define_macros = [
    ('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION'),
]

# Extension module
ext_modules = [
    Extension(
        'pyafmm',
        sources=sources,
        include_dirs=include_dirs,
        define_macros=define_macros,
        language='c++',  # Still use c++ as the main language for pybind11
    ),
]

setup(
    name='pyafmm',
    version='0.1.0',
    author='Your Name',
    author_email='your.email@example.com',
    description='Python bindings for AFMM Skeletonization',
    long_description='',
    ext_modules=ext_modules,
    cmdclass={'build_ext': CustomBuildExt},
    zip_safe=False,
    python_requires='>=3.6',
    setup_requires=['pybind11>=2.5.0'],
    install_requires=[
        'numpy>=1.13.0',
        'pillow',
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3',
        'Programming Language :: C++',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: Image Processing',
    ],
)