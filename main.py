"""
All Rights Reserved

Authors:
  - Lucas Placentino
  - Chloé Blommaert
  - Numa Deville
"""

import pandas as pd
import numpy as np
import matplotlib as plt
import scipy as sp

# function to read data from xlsx file:
def read_excel_data(filename: str):
    # read data from xlsx file
    data = pd.read_excel(filename)
    return data


def main():
    # 1. Read data from OCV-SOC xlsx file
    # make lookup table from it - OR - create a fitting math func
    ocv_soc_data = read_excel_data("data/Cha_Dis_OCV_SOC_Data.xlsx")


    # -------------------------
    # 2. Model



    # -------------------------
    # 3. Optimization/Filtering
    # Extended Kalman Filter
    # Unscented Kalman Filter

    # Error adjustment


    pass

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(e)
    finally:
        print("###End program###")
