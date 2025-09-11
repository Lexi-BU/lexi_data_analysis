from spacepy.pycdf import CDF
import numpy as np 
import glob
from pathlib import Path
import matplotlib.pyplot as plt

folder_name = "../data/from_lexi/2024/processed_data/sci/level_1c/cdf/1.0.0/"

folder_name = Path(folder_name).expanduser().resolve()

file_name_list = np.sort(glob.glob(folder_name + "/*.cdf"))
