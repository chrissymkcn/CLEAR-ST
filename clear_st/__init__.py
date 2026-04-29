import os
# Fix Numba caching issue before any other imports
# os.environ['NUMBA_DISABLE_CACHING'] = '1'
# os.environ['NUMBA_CACHE_DIR'] = f'{os.environ["HOME"]}/tmp'

__version__ = '0.1'

from SPUndiff.undiff import undiff
from SPUndiff.clear_model import CLEARmodel

__all__ = [
    'undiff',
    'CLEARmodel',
]