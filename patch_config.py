import os
import pathlib
original_mkdir = pathlib.Path.mkdir
def patched_mkdir(self, *args, **kwargs):
    if str(self).startswith("/Volumes"):
        return
    return original_mkdir(self, *args, **kwargs)
pathlib.Path.mkdir = patched_mkdir
