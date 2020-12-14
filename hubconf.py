# -*- coding: utf-8 -*- #
"""************************************************************************************************
   FileName     [ hubconf.py ]
   Synopsis     [ interface to Pytorch Hub: https://pytorch.org/docs/stable/hub.html#torch-hub ]
   Author       [ S3PRL ]
   Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
************************************************************************************************"""

import os
import hashlib
import pathlib
import importlib

import torch
dependencies = ['torch']


hubconf_search_root = '.'
hubconfs = [str(p) for p in pathlib.Path(hubconf_search_root).rglob('hubconf.py')]
hubconfs.remove('hubconf.py')  # remove the root hubconf.py

for hubconf in hubconfs:
    module_name = '.'.join(str(hubconf).split('.')[:-1]).replace('/', '.')
    _module = importlib.import_module(module_name)
    for variable_name in dir(_module):
        _variable = getattr(_module, variable_name)
        if callable(_variable) and variable_name[0] != '_':
            globals()[variable_name] = _variable
