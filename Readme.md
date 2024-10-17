# Team7 - Huawei 2024 Nuremberg Tech Arena
Lithium-Ion Battery State of Charge Challenge  

Need good/high/best efficiency (good for embedded use) -> **avoid** neural networks, use system that can easily be made into an FPGA or ASIC  
We'll use an empirical model (along with an electrical model ?) -> provided by the challenge  
1. (off-line) _OCV_-SOC relationship is predetermined  -> stored in lookup table or fitted by a math func
2. (off-line) Get model parameters
3. (on-line) Filtering to better estimation
> Wei, J., Dong, G., & Chen, Z. (2017). On-board adaptive model for state of charge estimation of lithium-ion batteries based on Kalman filter with proportional integral-based error adjustment. Journal of Power Sources, 365, 308-319.  

- Ampere hours integral (AHI) -> needs open circuit voltage (_OCV_)  
- Kalman filter (regular, (adaptive?) extended, (adaptive?) unscented, sigma-point ?) + regulator/observer (P, PI, PD or PID ?) for error correction/adjustment -> exteded KF because non-linear  
> Wang, W., & Mu, J. (2019). State of charge estimation for lithium-ion battery in electric vehicle based on Kalman filter considering model error. Ieee Access, 7, 29223-29235.
