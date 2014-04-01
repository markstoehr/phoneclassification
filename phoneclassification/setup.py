import os
from os.path import join
from _build_utils import get_blas_info

def configuration(parent_package='', top_path=None):
    import numpy
    from numpy.distutils.misc_util import Configuration
    
    cblas_libs, blas_info = get_blas_info()
    cblas_compile_args = blas_info.pop('extra_compile_args', [])
    cblas_includes = [join('..', 'src', 'cblas'),
                      numpy.get_include(),
                      blas_info.pop('include_dirs', [])]



    config = Configuration('', parent_package, top_path)
    if os.name == 'posix':
        cblas_libs.append('m')

    config.add_extension('multicomponent_binary_sgd',
                         sources=['multicomponent_binary_sgd.c'],
                         include_dirs=cblas_includes,
                         libraries=cblas_libs,
                         **blas_info)

    config.add_extension('multicomponent_lowrank_svm',
                         sources=['multicomponent_lowrank_svm.c'],
                         include_dirs=cblas_includes,
                         libraries=cblas_libs,
                         **blas_info)

    config.add_extension('svm',
                         sources=['svm.c'],
                         include_dirs=cblas_includes,
                         libraries=cblas_libs,
                         **blas_info)
    config.add_extension('_fast_EM',
                         sources=['_fast_EM.c'],
                         include_dirs=cblas_includes,
                         libraries=cblas_libs,
                         **blas_info)
    
    
    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())

