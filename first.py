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
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] :\n%(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
log = logging.getLogger(__name__)

# function to read data from xlsx file:
def read_excel_data(filename: str, **kwargs) -> pd.DataFrame:
    # read data from xlsx file
    data = pd.read_excel(filename, **kwargs)
    return data


def main():
    # 1. Read data from OCV-SOC xlsx file
    # make lookup table from it - OR - create a fitting math func
    charging_ocv_soc_data: pd.DataFrame = read_excel_data("data/Cha_Dis_OCV_SOC_Data.xlsx", usecols="B:C", skiprows=1, names=["SOC", "OCV"])
    log.debug(charging_ocv_soc_data)
    discharging_ocv_soc_data: pd.DataFrame = read_excel_data("data/Cha_Dis_OCV_SOC_Data.xlsx", usecols="E:F", skiprows=1, names=["SOC", "OCV"])
    log.debug(discharging_ocv_soc_data)
    

    #charging_ocv_soc_data.plot(x="SOC", y="OCV", kind="line", title="Charging OCV-SOC data")
    #discharging_ocv_soc_data.plot(x="SOC", y="OCV", kind="line", title="Discharging OCV-SOC data")



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
