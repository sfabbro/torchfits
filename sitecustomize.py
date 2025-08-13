# sitecustomize: executed automatically on interpreter startup if on sys.path
import os, sys
if sys.platform == 'darwin' and 'KMP_DUPLICATE_LIB_OK' not in os.environ:
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
