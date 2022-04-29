import copy
import json
import logging
import os
import pickle
import re
import shutil
import threading
from pathlib import Path

import torch
from alfred import constants
from alfred.data.preprocessor import Preprocessor
from alfred.nn.enc_visual import FeatureExtractor
from alfred.utils import data_util, helper_util, model_util
from progressbar import ProgressBar
from sacred import Experiment, Ingredient
from vocab import Vocab