import importlib
from pathlib import Path

import lexi_data_analysis_functions as lexi_functions
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

importlib.reload(lexi_functions)


input_dict = {
    "x_key": "x_volt_lin",
    "y_key": "y_volt_lin",
    "start_time": "2025-03-06T15:15:00Z",
    "end_time": "2025-03-06T15:45:00Z",
    "bins": 200,
    "bin_range": [-0.1, 0.1, -0.1, 0.1],
    "time_normalization": True,
}

df = 