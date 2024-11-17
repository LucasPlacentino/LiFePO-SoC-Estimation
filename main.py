import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import timeit
import statistics

start = timeit.default_timer()

test_str = str(input("Run test? (Y/n) ")).lower()
TEST = True if test_str == "y" else False
print(f"Running in {"test mode" if TEST else "prod mode"}")


print("Getting data from OCV-SOC file...")
#data_ocv_charge = pd.read_csv("charging_OCV_curve.csv", sep=';', header=None)
data_ocv_charge = pd.read_excel("data/Cha_Dis_OCV_SOC_Data.xlsx", header=1, usecols="B:C", skiprows=0) # voir les trucs à rajouter en arg de read_excel
data_ocv_charge.columns = ['data_SOC', 'data_U']
print("data_ocv_charge (head):")
print(data_ocv_charge.head())

#data_ocv_discharge = pd.read_csv("discharging_OCV_curve.csv", sep=';', header=None)
data_ocv_discharge = pd.read_excel("data/Cha_Dis_OCV_SOC_Data.xlsx", header=1, usecols="E:F", skiprows=0) # voir les trucs à rajouter en arg de read_excel
data_ocv_discharge.columns = ['data_SOC', 'data_U']
print("data_ocv_discharge (head):")
print(data_ocv_discharge.head())


"""
# We don't use the hppc test right now, as impedance can be (for now) estimated as a constant value (see plots)
print("Getting data from HPPC files...")
data_hppc_chg = pd.read_excel("data/EVE_HPPC_1_25degree_CHG-injectionTemplate.xlsx", header=3, usecols="A:D")
data_hppc_chg.columns = ['seconds', 'voltage', 'current_inv', 'SOC_true']
print("data_hppc_chg (head):")
print(data_hppc_chg.head())
data_hppc_dsg = pd.read_excel("data/EVE_HPPC_1_25degree_DSG-injectionTemplate.xlsx", header=3, usecols="A:D")
data_hppc_dsg.columns = ['seconds', 'voltage', 'current_inv', 'SOC_true']
print("data_hppc_dsg (head):")
print(data_hppc_dsg.head())
"""

#input("Press Enter to continue...")

if TEST:
    print("Reading test data...")
    # Dataset files are in "./data/Scenario-{nb}/{filename}.xlsx"
    # the directory "./data/" is in the gitignore to prevent uploading the confidential dataset to the repo
    files = [
        "data/Scenario-1/GenerateTestData_S1_Day0to4.xlsx",
        "data/Scenario-1/GenerateTestData_S1_Day4to7.xlsx",
        "data/Scenario-2/GenerateTestData_S2_Day0to4.xlsx",
        "data/Scenario-2/GenerateTestData_S2_Day4to7.xlsx",
        "data/Scenario-3/GenerateTestData_S3_Day0to4.xlsx",
        "data/Scenario-3/GenerateTestData_S3_Day4to7.xlsx",
        "data/Scenario-4/GenerateTestData_S4_Day0to4.xlsx",
        "data/Scenario-4/GenerateTestData_S4_Day4to7.xlsx"
    ]
    # add additional files to the list here above

    for i in range(len(files)):
        print(f"{i}. {files[i]}")
    data_choice: int = int(input(f"Select data file: "))

    print("Reading data from file:", files[data_choice],"...")
    # read file with these (hardcoded) specific columns placement and headers:
    data = pd.read_excel(files[int(data_choice)], usecols="B:D,G", skiprows=[0,3],header=1) # read file situated in ./data/Scenario-{nb}/{filename}.xlsx
else:
    print("Reading ./test.xlsx file... (THE SPREADSHEET FORMATTING MUST BE THE SAME AS THE PROVIDED TEST DATA)")
    #data = pd.read_csv("test.csv", sep=';', header=None) # ? sep correct ?
    #data = pd.read_csv("test.csv", header=None) # ?
    data = pd.read_excel("test.xlsx", usecols="B:D,G", skiprows=[0,3],header=1)
    # same columns format as test provided data


data.columns = ['voltage', 'current_inv', 'SOC_true', 'temperature']
print(data.head()) # view the first few lines of the dataframe to manually verify the data has been read correctly
#input("Run algorithm? Press Enter to continue...")

# Extract columns by index
voltage_data = data['voltage'].values
current_data = data['current_inv'].values
soc_true_data = data['SOC_true'].values
temperature_data = data['temperature'].values

soc_ocv_charge = data_ocv_charge["data_SOC"].values
voltage_ocv_charge = data_ocv_charge["data_U"].values
R_int_charge = 0.06 #R_int_chg

soc_ocv_discharge = data_ocv_discharge["data_SOC"].values
voltage_ocv_discharge = data_ocv_discharge["data_U"].values
R_int_discharge = 0.06 #R_int_dsg #! negative ?

# Battery parameters
nominal_capacity = 280.0  # Nominal capacity in Ah (adjust based on the battery)
dt = 1.0  # Time step in seconds
initial_SoC = soc_true_data[0]  # Initial SoC

# Kalman filter initialization
SoC_est = np.array([[initial_SoC]])
P = np.array([[50]])
Q = np.array([[1e-20]])
R = np.array([[0.1]])

# Voltage model using charge and discharge curves
def voltage_model(SOC, current, temperature=None, mode="discharge"):
    if mode == "charge":
        soc_ocv = soc_ocv_charge
        voltage_ocv = voltage_ocv_charge
        R_int = R_int_charge
    else:
        soc_ocv = soc_ocv_discharge
        voltage_ocv = voltage_ocv_discharge
        R_int = R_int_discharge

    # Voltage interpolation based on SoC
    V_oc = np.interp(SOC, soc_ocv, voltage_ocv)

    # Adjustment according to the specific internal resistance
    return V_oc - current * R_int

# Jacobian of the state transition (approximation for the SoC)
def jacobian_state_transition():
    return np.array([[1]])

# Jacobian of the measurement function
def jacobian_measurement_function():
    return np.array([[1]])

# Initialization for maximum error calculation
max_ae = 0

# Simulation loop
num_steps = len(voltage_data)
print(f"Total number of steps : {num_steps}")
SoC_values = []
error =[]
OffSet = soc_true_data[0]

for t in range(num_steps):
    current = current_data[t]
    measured_voltage = voltage_data[t]
    #temperature = temperature_data[t]

    # Prediction
    SoC_pred = SoC_est + np.array([[current * dt / nominal_capacity]])
    F = jacobian_state_transition()
    P_pred = F @ P @ F.T + Q

    # Predicted voltage (choose between charge/discharge based on current)
    mode = "charge" if current > 0 else "discharge" # > or < ?
    voltage_pred = voltage_model(SoC_pred[0, 0], current, mode=mode)

    # Update (filtering)
    H = jacobian_measurement_function()
    K = P_pred @ H.T @ np.linalg.inv(H @ P_pred @ H.T + R)
    innovation = measured_voltage - voltage_pred
    SoC_est = SoC_pred + K @ np.array([[measured_voltage - voltage_pred]])
    P = (np.eye(len(P)) - K @ H) @ P_pred

    # Store the estimated SoC value
    SoC_values.append(SoC_est[0, 0])

    # Calculate the maximum absolute error
    actual_soc = soc_true_data[t]
    error.append(abs(actual_soc/40 + OffSet - SoC_est[0, 0]) / SoC_est[0,0])

# Display results

print('max estimated soc', max(SoC_values))
print('max real soc', max(soc_true_data))

stop = timeit.default_timer()
print('Time: ', stop - start)

SoC_values = np.array(SoC_values)/40 + OffSet

error = abs(statistics.mean(error))
print("error", abs(1-error)*100, ' %')


if TEST:
    np.savetxt("SoC_values_estim.csv", SoC_values, delimiter=",")
    np.savetxt("SoC_values_real.csv", soc_true_data, delimiter=",")
    print("Plotting...")
    plt.plot(range(num_steps), SoC_values, label="Estimated SoC")
    plt.plot(range(num_steps), soc_true_data, label="Real SoC")
    plt.xlabel("Time (s)")
    plt.ylabel("State of Charge (SoC)")
    plt.title(f"SoC Estimation with Extended Kalman Filter (Scenario {data_choice})")
    plt.legend()
    plt.show()
else:
    np.savetxt("out.csv", SoC_values, delimiter=",")
    # save as excel file
    df = pd.DataFrame(SoC_values)
    df.to_excel("out.xlsx", index=False, header=False)