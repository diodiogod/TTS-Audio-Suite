#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   __init__.py.py    
@Contact :   haoheliu@gmail.com
@License :   (C)Copyright 2020-2100

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
9/14/21 1:00 AM   Haohe Liu      1.0         None
"""

import os
from .config import Config

# Note: Model path check is now deferred to the node's initialization
# This allows the node to patch Config.ckpt BEFORE importing voicefixer_bundled
# If Config.ckpt is not set properly by the node, it will fail at vocoder initialization time
