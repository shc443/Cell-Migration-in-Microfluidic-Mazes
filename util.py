#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
import fire
import logging
from tqdm import tqdm
from collections import Counter
import pathlib
import pickle
import json
import glob

import os

#import dask.dataframe as dd
from tqdm.auto import tqdm
tqdm.pandas()
ProgressBar().register()
from util import Helper
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math
import json
import os

class Helper:
    
    def __init__(self, config_path=os.path.dirname(os.path.abspath("__file__"))+'/config.json'):
      
        with open(config_path) as json_file:
            data = json.load(json_file)
            
        curnt_dir = os.path.dirname(os.path.abspath("__file__"))
        
        data_path = data['param']
        self.N0 = curnt_dir+data_path['N0']
        self.C0 = curnt_dir+data_path['C0']
        self.Uc = curnt_dir+data_path['Uc']
        self.Un = curnt_dir+data_path['Un']
        self.Dc = curnt_dir+data_path['Dc']
        self.Qcb0 = curnt_dir+data_path['Qcb0']
        self.Qcd0 = curnt_dir+data_path['Qcd0']
        self.Qn = curnt_dir+data_path['Qn']
        self.A0 = curnt_dir+data_path['A0']
        self.L_ = curnt_dir+data_path['L_']
        self.M = curnt_dir+data_path['M']
        self.W = curnt_dir+data_path['W']
        self.H = curnt_dir+data_path['H']
        self.dx = curnt_dir+data_path['dx']
        self.dt = curnt_dir+data_path['dt']
        self.M = curnt_dir+data_path['M']
        
        save_path = data['save_path']
        self.plot_conc = save_path['plot_conc_path']
#        self.plot_conc = save_path['plot_conc_path']
#        self.plot_conc = save_path['plot_conc_path']
        
        
if __name__ == "__main__":
    fire.Fire(Helper)
