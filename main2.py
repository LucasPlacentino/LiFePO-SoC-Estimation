import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import timeit
import statistics
import csv

start = timeit.default_timer()


# Since we need to use python, the code is "rather slow". For a real application of this program like an embedded
# system, we would use C++ or Rust for better performance, with compiler optimizations enabled which would render
# the code much (MUCH) faster, probably multiple orders of magnitude faster. (Espescially since we store the
# entire dataset in memory, where real-world applications would read data continuously from a sensor processing
# it in real-time)


test_str = str(input("Run DEBUG? This will use matplotlib to plot data, use with UI! (Y/n) ")).lower()
TEST = True if test_str == "y" else False
print(f"Running in {"debug mode" if TEST else "prod mode"}")


# We read static data from the OCV-SOC and HPPC files during runtime for simplicity in this simulation,
# but in a real-world application, this data would be read from direct memory like a lookup table.

print("Getting data from OCV-SOC file...")
#data_ocv_charge = pd.read_csv("charging_OCV_curve.csv", sep=';', header=None)
data_ocv_charge = pd.read_excel("data/Cha_Dis_OCV_SOC_Data.xlsx", header=1, usecols="B:C", skiprows=0)
data_ocv_charge.columns = ['data_SOC', 'data_U']
print("data_ocv_charge (head):")
print(data_ocv_charge.head())

#data_ocv_discharge = pd.read_csv("discharging_OCV_curve.csv", sep=';', header=None)
data_ocv_discharge = pd.read_excel("data/Cha_Dis_OCV_SOC_Data.xlsx", header=1, usecols="E:F", skiprows=0)
data_ocv_discharge.columns = ['data_SOC', 'data_U']
print("data_ocv_discharge (head):")
print(data_ocv_discharge.head())


"""
# We don't use the HPPC test right now, as impedance can be (for now) estimated as a constant value
# (see result plots from HPPC tests)

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

time_read = timeit.default_timer()
print(f"Time to read data: {time_read - start} s")

# Extracting columns by index
voltage_data = data['voltage'].values
current_data = data['current_inv'].values
soc_true_data = data['SOC_true'].values
temperature_data = data['temperature'].values

# Battery parameters
nominal_capacity = 280.0  # Nominal capacity in Ah (adjust based on the battery)
dt = 1.0  # Time step in seconds
initial_voltage = voltage_data[0]
print(f"Initial voltage: {initial_voltage}")

initial_SoC = soc_true_data[0]

"""
# We could find the initial SoC by interpolating the initial voltage from the OCV-SOC curves
# or find the closest. This did not work well in practice, so we use the initial SoC from the data
initial_current = np.mean(current_data[0:10])
if initial_current < 0:
    ocv_SOC_dis = data_ocv_discharge["data_SOC"].values
    ocv_U_dis = data_ocv_discharge["data_U"].values
    #initial_SoC = np.interp(initial_voltage, ocv_U_dis, ocv_SOC_dis)
    index_closest_voltage_dis = np.argmin(np.abs(ocv_U_dis - initial_voltage))
    closest_SoC_dis = ocv_SOC_dis[index_closest_voltage_dis]
    initial_SoC_dis = closest_SoC_dis
    print(f"Initial SoC DIS: {initial_SoC_dis}")
    initial_SoC = initial_SoC_dis
else:
    ocv_SOC_chg = data_ocv_charge["data_SOC"].values
    ocv_U_chg = data_ocv_charge["data_U"].values
    #initial_SoC = np.interp(initial_voltage, ocv_U_chg, ocv_SOC_chg)
    index_closest_voltage_chg = np.argmin(np.abs(ocv_U_chg - initial_voltage))
    closest_SoC_chg = ocv_SOC_chg[index_closest_voltage_chg]
    initial_SoC_chg = closest_SoC_chg
    print(f"Initial SoC CHG: {initial_SoC_chg}")
    initial_SoC = initial_SoC_chg
# OR take the mean of the two SoC values (charge and discharge)
#initial_SoC = (initial_SoC_dis + initial_SoC_chg) / 2
"""

#initial_SoC = 50 # TEST

print(f"Initial SoC: {initial_SoC}, true initial SoC: {soc_true_data[0]}")

# Kalman filter initialization
SoC_est = np.array([[initial_SoC]])
P = np.array([[1]])     # tune this value
Q = np.array([[1e-10]]) # tune this value
R = np.array([[0.05]])  # tune this value

#OffSet = soc_true_data[0]
OffSet = 0
factor = -36.12201 # -30.82 # -35.341 # -36.12201


# Voltage model using charge and discharge curves
def voltage_model(SOC, current, temperature=None, mode="discharge"): # temp not used YET
    if mode == "charge":
        soc_ocv = data_ocv_charge["data_SOC"].values
        voltage_ocv = data_ocv_charge["data_U"].values
        R_int = 0.06 #R_int_chg
        # todo: use hppc test data to get R_int for current SoC
        # constant R value is sufficiently accurate for now
        #R_int = impedance_from_hppc_charge(SOC)
    else:
        soc_ocv = data_ocv_discharge["data_SOC"].values
        voltage_ocv = data_ocv_discharge["data_U"].values
        R_int = -0.01 #R_int_dsg
        # todo: use hppc test data to get R_int for current SoC
        # constant R value is sufficiently accurate for now

    V_oc = np.interp(SOC, soc_ocv, voltage_ocv)
    return V_oc - current * R_int

# Jacobian of state transition (approximation for SoC)
def jacobian_state_transition():
    return np.array([[1]])

## We'll try this one out later :
#def jacobian_state_transition(current, dt, nominal_capacity):
#    # For simplicity, assuming a linear model with respect to current
#    # but could include a term for temperature or resistance changes
#    jacobian = np.array([[1 - (current * dt / nominal_capacity)]])
#    return jacobian

# Jacobian of the measurement function
#def jacobian_measurement_function(*args, **kwargs):
#    return np.array([[1]])

def jacobian_measurement_function(SoC, current, delta=1e-5):
    # Numerical differentiation for dV_pred / dSoC
    ## We could try to use sympy.diff() to calculate this jacobian
    ## rather than using numerical differentiation (would be better performance wise ?)
    V_plus = voltage_model(SoC + delta, current)
    V_minus = voltage_model(SoC - delta, current)
    dV_dSoC = (V_plus - V_minus) / (2 * delta)
    return np.array([[dV_dSoC]])

# Initialization for maximum error calculation
max_ae = 0

# Simulation loop
num_steps = len(voltage_data)
print(f"Total nb of steps : {num_steps}")
SoC_values = []
errors = []

for t in range(num_steps):
    current = current_data[t]/factor # -30.82 # -35.341
    #current = np.mean(current_data[:5]) / -30.82
    measured_voltage = voltage_data[t]
    temperature = temperature_data[t]

    # Prediction
    SoC_pred = SoC_est - np.array([[current * dt / nominal_capacity]])
    F = jacobian_state_transition()
    P_pred = F @ P @ F.T + Q

    # Predicted voltage (choose charge/discharge mode based on current)
    mode = "charge" if current > 0 else "discharge"
    voltage_pred = voltage_model(SoC_pred[0, 0], current, mode=mode)

    # Update (filtering)
    H = jacobian_measurement_function(SoC_pred[0, 0], current)
    K = P_pred @ H.T @ np.linalg.inv(H @ P_pred @ H.T + R + 1e-9 * np.eye(H.shape[0]))
    innovation = measured_voltage - voltage_pred
    SoC_est = SoC_pred + K @ np.array([[measured_voltage - voltage_pred]])
    P = (np.eye(len(P)) - K @ H) @ P_pred

    # Store the estimated SoC value
    SoC_values.append(SoC_est[0, 0])

    # Calculate maximum absolute error
    actual_soc = soc_true_data[t]

    errors.append(actual_soc - SoC_est[0, 0])

    absolute_error = abs(SoC_est[0, 0] - actual_soc)
    if absolute_error > max_ae:
        max_ae = absolute_error



# Displaying results

print('max soc estim', max(SoC_values))
print('max true soc', max(soc_true_data))


# Display maximum error
print(f"Maximum Absolute Error (MaxAE) : {max_ae} %")

SoC_values = np.array(SoC_values) #+ OffSet

squared_errors = [(actual_soc + OffSet - est_soc)**2 for actual_soc, est_soc in zip(soc_true_data, SoC_values)] # list of squared errors # yo the zip() function is actually so cool so useful
mse = sum(squared_errors) / len(squared_errors)  # Mean Squared Error
rmse = np.sqrt(mse) # Root Mean Squared Error
print(f"Root Mean Square Error (RMSE): {rmse} %")

stop = timeit.default_timer()

print('Time: ', stop - time_read, 's')

if TEST:
    plt.plot(range(num_steps), SoC_values, label="SoC Estimation")
    plt.plot(range(num_steps), soc_true_data, label="Real SoC")
    # make the y axis of the plot go from 0 to 100:
    plt.ylim(0, 100)
    plt.xlim(0, num_steps)
    plt.grid()
    plt.xlabel("Time [s]")
    plt.ylabel("State of Charge (SoC) [%]")
    plt.title("SoC Estimation with Extended Kalman Filter")
    plt.legend()
    plt.show()

    time_plot = timeit.default_timer()
    print('Plot time: ', time_plot - stop, 's')

    # write to out.xlsx file (along with the rest of the initial test.xlsx file) column "Unnamed: 4" the values in SoC_values and column "Unnamed: 5" the errors
    print("Saving SoC_values to out.xlsx...")
    print("Copying data file to out.xlsx")
    out = pd.read_excel(files[int(data_choice)])
    print(out.head())
    print("Adding SoC_values and errors to out.xlsx")
    out.loc[3:3+len(SoC_values)-1, 'Unnamed: 4'] = SoC_values
    out.loc[3:3+len(errors)-1, 'Unnamed: 5'] = errors
    out.columns = ['Name', '', '', '', '', '', '', '']
    if bool(input("Save to FILE out.xlsx? (Y/n) ").lower() == "y"):
        print("Saving out.xlsx")
        out.to_excel("out.xlsx", index=False, header=True)
    print(out.head())
    print(out.tail())

    time_save = timeit.default_timer()
    print('Save time: ', time_save - time_plot, 's')

else:
    # write to out.xlsx file (along with the rest of the initial test.xlsx file) column "Unnamed: 4" the values in SoC_values and column "Unnamed: 5" the errors
    print("Saving SoC_values to out.xlsx...")
    print("Copying data file to out.xlsx")
    out = pd.read_excel("test.xlsx")
    print("Adding SoC_values and errors to out.xlsx")
    out.loc[3:3+len(SoC_values)-1, 'Unnamed: 4'] = SoC_values
    out.loc[3:3+len(errors)-1, 'Unnamed: 5'] = errors
    print("Saving out.xlsx")
    out.columns = ['Name', '', '', '', '', '', '', '']
    out.to_excel("out.xlsx", index=False, header=True)

    time_save = timeit.default_timer()
    print('Save time: ', time_save - stop, 's')

print("Done.")
