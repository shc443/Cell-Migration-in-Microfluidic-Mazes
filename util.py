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

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from konlpy.tag import Mecab
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF, PCA
from gensim.test.utils import common_texts
from gensim.corpora.dictionary import Dictionary
from gensim.models.nmf import Nmf as GensimNMF
from gensim.models.coherencemodel import CoherenceModel
from sklearn.preprocessing import MinMaxScaler, normalize
from gensim.models.word2vec import Word2Vec
import os

#import dask.dataframe as dd
from dask.multiprocessing import get
from dask.diagnostics import ProgressBar
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
        
        self.N0 = data_path['N0']
        self.C0 = data_path['C0']
        self.Uc = data_path['Uc']
        self.Un = data_path['Un']
        self.Dc = data_path['Dc']
        self.Qcb0 = data_path['Qcb0']
        self.Qcd0 = data_path['Qcd0']
        self.Qn = data_path['Qn']
        self.A0 = data_path['A0']

if __name__ == "__main__":
    fire.Fire(Helper)
