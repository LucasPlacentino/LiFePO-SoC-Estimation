import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import timeit

start = timeit.default_timer()


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

# Dataset files are in "./data/Scenario-{nb}/{filename}.xlsx"
# the directory "./data/" is in the gitignore to prevent uploading the confidential dataset to the repo
files = [
    "Scenario-1/GenerateTestData_S1_Day0to4.xlsx",
    "Scenario-1/GenerateTestData_S1_Day4to7.xlsx",
    "Scenario-2/GenerateTestData_S2_Day0to4.xlsx",
    "Scenario-2/GenerateTestData_S2_Day4to7.xlsx",
    "Scenario-3/GenerateTestData_S3_Day0to4.xlsx",
    "Scenario-3/GenerateTestData_S3_Day4to7.xlsx",
    "Scenario-4/GenerateTestData_S4_Day0to4.xlsx",
    "Scenario-4/GenerateTestData_S4_Day4to7.xlsx"
]
# add additional files to the list here above

for i in range(len(files)):
    print(f"{i}. {files[i]}")
data_choice: int = int(input(f"Select data file: "))

print("Reading data from file:", files[data_choice],"...")
# read file with these (hardcoded) specific columns placement and headers:
data = pd.read_excel("data/"+files[int(data_choice)], usecols="B:D,G", skiprows=[0,3],header=1) # read file situated in ./data/Scenario-{nb}/{filename}.xlsx
data.columns = ['voltage', 'current_inv', 'SOC_true', 'temperature']
print(data.head()) # look the first few lines of the dataframe to manually verify the data has been read correctly
#input("Run algorithm? Press Enter to continue...")

print('Reading Excel file took ', timeit.default_timer() - start, 's')


# Extraction des colonnes par index
voltage_data = data['voltage'].values
current_data = data['current_inv'].values
soc_true_data = data['SOC_true'].values
temperature_data = data['temperature'].values

soc_ocv_charge = data_ocv_charge["data_SOC"].values
voltage_ocv_charge = data_ocv_charge["data_U"].values
R_int_charge = 0.06 #R_int_chg

soc_ocv_discharge = data_ocv_discharge["data_SOC"].values
voltage_ocv_discharge = data_ocv_discharge["data_U"].values
R_int_discharge = -0.02 #R_int_dsg #! negative ?

# Paramètres de la batterie
nominal_capacity = 280.0  # Capacité nominale en Ah (ajuster en fonction de la batterie)
dt = 1.0  # Pas de temps en secondes
initial_SoC = soc_true_data[0]  # SoC initial
#initial_SoC = 90 # test trensient

# Initialisation du filtre de Kalman
SoC_est = np.array([[initial_SoC]])
# EKF parameters :
P = np.array([[50]])
Q = np.array([[1e-20]])
R = np.array([[1e-1]])


# Modèle de tension utilisant les courbes de charge et décharge
def voltage_model(SOC, current, temperature=None, mode="discharge"):
    if mode == "charge":
        soc_ocv = soc_ocv_charge
        voltage_ocv = voltage_ocv_charge
        R_int = R_int_charge
    else:
        soc_ocv = soc_ocv_discharge
        voltage_ocv = voltage_ocv_discharge
        R_int = R_int_discharge

    # Interpolation de la tension en fonction du SoC
    V_oc = np.interp(SOC, soc_ocv, voltage_ocv)

    # Ajustement selon la résistance interne spécifique
    return V_oc - current * R_int


# Jacobienne de la transition d'état (approximation pour le SoC)
def jacobian_state_transition():
    return np.array([[1]])


# Jacobienne de la mesure
#def jacobian_measurement_function(*args, **kwargs):
#    return np.array([[1]])

def jacobian_measurement_function(SoC, current, delta=1e-5):
    # Numerical differentiation for dV_pred / dSoC
    V_plus = voltage_model(SoC + delta, current)
    V_minus = voltage_model(SoC - delta, current)
    dV_dSoC = (V_plus - V_minus) / (2 * delta)
    return np.array([[dV_dSoC]])

# Initialisation pour le calcul de l'erreur maximale
max_ae = 0

# Boucle de simulation
num_steps = len(voltage_data)
print(f"Steps: {num_steps}")
SoC_values = []

for t in range(num_steps):
    current = current_data[t]
    measured_voltage = voltage_data[t]
    #temperature = temperature_data[t]

    # Prédiction
    SoC_pred = SoC_est - np.array([[current * dt / (nominal_capacity)]]) # nominal_capacity in Ah, dt in s -> *3600 to convert Ah to As ?
    F = jacobian_state_transition()
    P_pred = F @ P @ F.T + Q

    # Tension prédite (choix entre charge/décharge selon le courant)
    mode = "charge" if current > 0 else "discharge" # > or < ?
    voltage_pred = voltage_model(SoC_pred[0, 0], current, mode=mode)

    # Mise à jour (filtrage)
    #H = jacobian_measurement_function()
    H = jacobian_measurement_function(SoC_pred[0, 0], current)
    K = P_pred @ H.T @ np.linalg.inv(H @ P_pred @ H.T + R) # +1e-9 to avoid singular matrix
    innovation = measured_voltage - voltage_pred # difference between measured and predicted voltage
    SoC_est = SoC_pred + K @ np.array([[innovation]]) # clip within 0 and 1 or 0 and 100 ?
    P = (np.eye(len(P)) - K @ H) @ P_pred

    # Stocker la valeur estimée de SoC
    SoC_values.append(SoC_est[0, 0])

    # Calcul de l'erreur absolue maximale
    actual_soc = soc_true_data[t]
    max_ae = max(max_ae, abs(SoC_est[0, 0] - actual_soc))
print("loop done.")

max_SoC_values = max(SoC_values)
min_SoC_values = min(SoC_values)

factor = int(max_SoC_values / max(soc_true_data))
factor = 57

# Affichage des résultats
#SoC_values += initial_SoC  # Ajouter le SoC initial
SoC_values = np.array(SoC_values)/factor
#SoC_values /= max_SoC_values  # Normaliser
#SoC_values *= 100  # Convertir en pourcentage
SoC_values *= -1  # Inverser pour correspondre aux données réelles
SoC_values += initial_SoC  # Ajouter le SoC initial

# error = sum((SoC_values-soc_true_data)/SoC_values)/len(SoC_values)
#print('error : ', error*100, ' %')

print('max soc estim', max_SoC_values/factor *(-1) + initial_SoC)
print('max vrai soc', max(soc_true_data))
print('min soc estim', min_SoC_values/factor *(-1) + initial_SoC)
print('min vrai soc', min(soc_true_data))

# Affichage de l'erreur maximale
print(f"Maximum Absolute Error (MaxAE): {max_ae/factor}")

#print(f'Root Mean Square Error (RMSE): {np.sqrt(np.mean((np.array(SoC_values) - np.array(soc_true_data))**2))}')

stop = timeit.default_timer()
print('Time: ', stop - start, 's')

plt.plot(range(num_steps), SoC_values, label="Estimated SoC")
plt.plot(range(num_steps), soc_true_data, label="True SoC")
plt.xlabel("Time (s)")
plt.ylabel("State Of Charge (SoC)")
plt.title("SoC estimation with an Extended Kalman filter")
plt.legend()
plt.show()

if input("Save SoC_values to a file? (y/n): ") == "y":
    # save SoC_values to a file
    np.savetxt("SoC_values.csv", SoC_values, delimiter=",")
