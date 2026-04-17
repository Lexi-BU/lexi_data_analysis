import pandas as pd
from pathlib import Path

LEXI_DATA_FILE = Path(
    "../data/line_profile_data/bg_corrected/from_l2/"
    "line_profile_fit_parameters_bg_corrected_1min.csv"
)
LEXI_HOUSEKEEPING_FILE = Path(
    "../data/lexi_hk_data_2025-01-16_00-00-00_to_2025-03-17_00-00-00_v0.0.csv"
)

lexi_hk_df_org = pd.read_csv(LEXI_HOUSEKEEPING_FILE)
lexi_hk_df_org.set_index("Epoch", inplace=True)

# Keep only the epoch and all_counts columns
lexi_hk_df_selected = lexi_hk_df_org[["DeltaEvntCount"]]

# Keep oly the data between 2025-03-16 19:30:00 and 2025-03-16 21:15:00
lexi_hk_df_selected = lexi_hk_df_selected.loc[
    "2025-03-16 19:30:00" : "2025-03-16 21:15:00"
]