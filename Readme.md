# Team7 - Huawei 2024 Nuremberg Tech Arena
Lithium-Ion Battery State of Charge Challenge  

## Project Description
The estimation of SoC in Lithium-Ion batteries is crucial for battery management systems, especially in electric vehicles, smartphones, and other portable devices. An accurate SoC estimation ensures optimal performance, prolongs battery life, and enhances user safety. Traditional methods often face challenges due to the non-linear characteristics of battery dynamics. This project leverages an **Extended Kalman Filter (EKF)** for real-time, accurate SoC estimation.  

We use an empirical model because we got the OCV-SOC relationship provided by the challenge. The model is based on the Open-Circuit Voltage (OCV) curve, which relates the SoC to the terminal voltage of the battery. The EKF algorithm handles the non-linear behavior of battery characteristics by linearizing the state and measurement equations around the current estimate. We also compute Jacobians for both the state transition and the measurement model to improve the estimation accuracy.  

## Code Structure
```plaintext
.
├── data/
│   ├── Scenario-1/                                        # Data files for different scenarios
|   |   ├── GenerateTestData_S1_Day0to4.xlsx
|   |   └── GenerateTestData_S1_Day4to7.xlsx
│   ├── Scenario-2/
|   |   ├── GenerateTestData_S2_Day0to4.xlsx
|   |   └── GenerateTestData_S2_Day4to7.xlsx
│   ├── Scenario-3/
|   |   ├── GenerateTestData_S3_Day0to4.xlsx
|   |   └── GenerateTestData_S3_Day4to7.xlsx
│   ├── Scenario-4/
|   |   ├── GenerateTestData_S4_Day0to4.xlsx
|   |   └── GenerateTestData_S4_Day4to7.xlsx
│   ├── Cha_Dis_OCV_SOC_Data.xlsx                          # OCV-SOC relationship
│   ├── EVE_HPPC_1_25degree_CHG-injectionTemplate.xlsx     # HPPC charging tests
|   └── EVE_HPPC_1_25degree_DSG-injectionTemplate.xlsx     # HPPC discharging tests
├── main.py                                                # Main script
├── README.md                                              # Project documentation
└── requirements.txt                                       # Python dependencies
```

## Usage
Run the `main.py` script with python and choose the excel file you would like to use for the analysis.

```bash
$ python main.py
Getting data from OCV-SOC file...
data_ocv_charge (head):
    [...]
data_ocv_discharge (head):
    [...]
0. Scenario-1/GenerateTestData_S1_Day0to4.xlsx
1. Scenario-1/GenerateTestData_S1_Day4to7.xlsx
2. Scenario-2/GenerateTestData_S2_Day0to4.xlsx
3. Scenario-2/GenerateTestData_S2_Day4to7.xlsx
4. Scenario-3/GenerateTestData_S3_Day0to4.xlsx
5. Scenario-3/GenerateTestData_S3_Day4to7.xlsx
6. Scenario-4/GenerateTestData_S4_Day0to4.xlsx
7. Scenario-4/GenerateTestData_S4_Day4to7.xlsx
Select data file: 
```

Input a number between 0 and 7, it will then give the plot and results after the computation.  

## Extended Kalman Filter (EKF) Approach
We use an EKF algorithm to estimate the SoC by linearizing the non-linear state and measurement equations around the current estimate. This method involves computing Jacobians for both the state transition and the measurement model.  

We prefered the EKF over other types of algorithm like neural networks because it is more suitable for embedded systems and can be easily implemented in a BMS (incorporated in an FPGA or even an ASIC). The EKF is also more efficient and requires less computational resources compared to neural networks.  

### Inspirations:
Papers:
> - Wei, J., Dong, G., & Chen, Z. (2017). On-board adaptive model for state of charge estimation of lithium-ion batteries based on Kalman filter with proportional integral-based error adjustment. Journal of Power Sources, 365, 308-319.  
> - Wang, W., & Mu, J. (2019). State of charge estimation for lithium-ion battery in electric vehicle based on Kalman filter considering model error. Ieee Access, 7, 29223-29235.  
> - Ahmed, M. S., Raihan, S. A., & Balasingam, B. (2020). A scaling approach for improved state of charge representation in rechargeable batteries. Applied energy, 267, 114880.  
> - El Maliki, A., Benlafkih, A., Anoune, K., & Hadjoudja, A. (2024). Reduce state of charge estimation errors with an extended Kalman filter algorithm. International Journal of Electrical and Computer Engineering (IJECE), 14(1), 57-65. doi:http://doi.org/10.11591/ijece.v14i1.pp57-65  


## Results

Maximum Absolute Error (MaxAE) : 0.10682889559265618 %
Root Mean Square Error (RMSE): 0.0624526869097305 %

<img width="641" alt="Capture d’écran 2024-11-17 à 22 57 52" src="https://github.com/user-attachments/assets/6962e9aa-88f8-43b8-ac79-aa60f8d95366">

## Improvements

### Fixes:
- **Algorithm**: there's some issue with the algorithm that needs to be fixed during the next phase in order to get a lower error.  
- **Battery Caracteristics**: the battery characteristics are defined as constants as it was considered as a sufficient simplification. It would be interesting to consider them as variables.

### Enhancements:
- **Modeling**: Implement a more accurate battery model to improve the estimation accuracy.  
- **Optimization**: Optimize the EKF algorithm to reduce the computational complexity and improve real-time performance.  
- **Validation**: Validate the algorithm on a wider range of scenarios and battery types to ensure robustness and generalization.  


## Acknowledgments
We would like to thank the organizers for hosting this challenge/hackathon, which enabled us to learn many new concepts and develop new skills.  

## Team 7 Members
- **Chloé Blommaert**  
- **Numa Deville**  
- **Lucas Placentino**  

> _All rights reserved._
