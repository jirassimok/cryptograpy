# Test init file, just sets up path.
import pathlib
import sys
try:
    sys.path.append(str(pathlib.Path(__file__).parent.parent))
finally:
    del sys, pathlib
