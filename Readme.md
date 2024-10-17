# Team7 - Huawei 2024 Nuremberg Tech Arena
Lithium-Ion Battery State of Charge Challenge  

Data from LiFePO4 Battery:
- capacity: 280Ah
- charge upper limit voltage: 3.65V
-  discharge lower threshold voltage: 2.5V
-  rated voltage: 3.2V
-  current rate range: 0~1C
-  rated current rate: 0.2C

data format: steps of 1 second

Criteria:
1. Accuracy: Maximum Absolute Error (MaxAE) measures the largest deviation between estimated and actual SoC values, assessing worst-case scenarios.
2. Robustness: Evaluates the algorithm's stability and accuracy across different conditions.
3. Efficiency: Assesses execution time, memory usage, and consistency across a uniform testing environment.
4. Transient Convergence: Measures the algorithm’s ability to quickly correct incorrect initial SoC values.
5. Documentation: Evaluates clarity, organization, and code quality.

Need good/high/best efficiency (good for embedded use) -> **avoid** neural networks, use system that can easily be made into an FPGA or ASIC, integratable into a BMS  
We'll use an empirical model (along with an electrical model ?) -> provided by the challenge  
1. (off-line) _OCV_-SOC relationship is predetermined  -> stored in lookup table or fitted by a math func
2. (off-line) Get model parameters
3. (on-line) Filtering to better estimation
> Wei, J., Dong, G., & Chen, Z. (2017). On-board adaptive model for state of charge estimation of lithium-ion batteries based on Kalman filter with proportional integral-based error adjustment. Journal of Power Sources, 365, 308-319.  

- Ampere hours integral (AHI) -> needs open circuit voltage (_OCV_)  
- Kalman filter (regular, (adaptive?) extended, (adaptive?) unscented, sigma-point, cubature ?) + regulator/observer (P, PI, PD or PID ?) for error correction/adjustment -> exteded KF because non-linear  
> Wang, W., & Mu, J. (2019). State of charge estimation for lithium-ion battery in electric vehicle based on Kalman filter considering model error. Ieee Access, 7, 29223-29235.

Assume that the open circuit voltage is error-free.  

> Ahmed, M. S., Raihan, S. A., & Balasingam, B. (2020). A scaling approach for improved state of charge representation in rechargeable batteries. Applied energy, 267, 114880.


> El Maliki, A., Benlafkih, A., Anoune, K., & Hadjoudja, A. (2024). Reduce state of charge estimation errors with an extended Kalman filter algorithm. International Journal of Electrical and Computer Engineering (IJECE), 14(1), 57-65. doi:http://doi.org/10.11591/ijece.v14i1.pp57-65

----------------------
Provided OCV-SOC relationship (100 datapoints x2: charge and discharge):  
![OCV-SOC_curve](https://github.com/user-attachments/assets/2c669ff4-10a9-4ecb-ac1b-cce6085051ea)


----------------------
HPPC (Hybrid pulse power characterization) charging and discharging tests ? 
