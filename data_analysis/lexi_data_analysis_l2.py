import datetime
import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd
from spacepy.pycdf import CDF as cdf

# Define the location of the L2 files
l2_files = sorted(glob.glob("./data/l2/*.cdf"))

start_time = datetime.datetime(2025, 3, 16, 19, 5)
end_time = datetime.datetime(2025, 3, 16, 19, 15)

# Find all the files in the time range
# Regex to capture timestamp in filename
pattern = re.compile(r"lexi_l2_(\d{12})")

# Select files in time range
selected_files = []
for f in l2_files:
    match = pattern.search(f)
    if match:
        ts = datetime.datetime.strptime(match.group(1), "%Y%m%d%H%M")
        if start_time <= ts <= end_time:
            selected_files.append(f)



