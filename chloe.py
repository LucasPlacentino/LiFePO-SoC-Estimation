import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

print("Getting data from OCV-SOC file...")
#data_ocv_charge = pd.read_csv("charging_OCV_curve.csv", sep=';', header=None)
data_ocv_charge = pd.read_excel("data/Cha_Dis_OCV_SOC_Data.xlsx", header=1, usecols="B:C", skiprows=0) # voir les trucs à rajouter en arg de read_excel
print("data_ocv_charge (head):")
print(data_ocv_charge.head())

#data_ocv_discharge = pd.read_csv("discharging_OCV_curve.csv", sep=';', header=None)
data_ocv_discharge = pd.read_excel("data/Cha_Dis_OCV_SOC_Data.xlsx", header=1, usecols="E:F", skiprows=0) # voir les trucs à rajouter en arg de read_excel
data_ocv_discharge.columns = ['data_SOC', 'data_U']
print("data_ocv_discharge (head):")
print(data_ocv_discharge.head())
input("Press Enter to continue...")

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
print(data.head()) # look the first few lines of the dataframe to manually verify the data has been read correctly
input("Press Enter to continue...")

# Dataframe has columns: "Voltage", "Current_inv", "SOC_true", and "Temp"


# Renommer les colonnes pour une meilleure lisibilité
#data_ocv_charge.columns = ['data_SOC', 'data_U']
#data_ocv_discharge.columns = ['data_SOC', 'data_U']


# Concaténer les deux fichiers #! Ne pas concaténer, on fait fichier par fichier
#data = pd.concat([data_days1_4, data_days4_7], ignore_index=True)

#print("Colonnes de data_ocv_charge:", data_ocv_charge.columns)
#print("Colonnes de data_ocv_discharge:", data_ocv_discharge.columns)

# Vérifier la présence des colonnes en affichant les 5 premières lignes
#print(data.head())

# Extraction des colonnes nécessaires en utilisant les index
# Supposons : voltage = colonne 0, current_inv = colonne 1, SOC_truc = colonne 2, temperature = colonne 3
#voltage_data = np.concatenate([data_days1_4.iloc[:, 0].values, data_days4_7.iloc[:, 0].values])  # Colonne 1
#current_data = np.concatenate([data_days1_4.iloc[:, 1].values, data_days4_7.iloc[:, 1].values])  # Colonne 2
#soc_truc_data = np.concatenate([data_days1_4.iloc[:, 2].values, data_days4_7.iloc[:, 2].values])  # Colonne 3
#temperature_data = np.concatenate([data_days1_4.iloc[:, 3].values, data_days4_7.iloc[:, 3].values])  # Colonne 4
# Extraction des colonnes par index
voltage_data = data['Voltage'].values
current_data = data['Current_inv'].values
soc_true_data = data['SOC_true'].values
temp_data = data['Temp'].values
# Paramètres de la batterie
nominal_capacity = 280.0  # Capacité nominale en Ah (ajuster en fonction de la batterie)
dt = 1.0  # Pas de temps en secondes
initial_SoC = soc_true_data[0]  # SoC initial

# Initialisation du filtre de Kalman
SoC_est = np.array([[initial_SoC]])
P = np.array([[1]])
Q = np.array([[1e-5]])
R = np.array([[0.1]])


# Modèle de tension utilisant les courbes de charge et décharge
def voltage_model(SOC, current, temperature, mode="discharge"):
    if mode == "charge":
        soc_ocv = data_ocv_charge["data_SOC"].values
        voltage_ocv = data_ocv_charge["data_U"].values
    else:
        soc_ocv = data_ocv_discharge["data_SOC"].values
        voltage_ocv = data_ocv_discharge["data_U"].values

    # Interpolation de la tension en fonction du SoC
    V_oc = np.interp(SOC, soc_ocv, voltage_ocv)

    # Ajustement selon la température et le courant
    R_int = 0.01  # Valeur de résistance interne (ajuster selon le modèle)
    return V_oc - current * R_int


# Jacobienne de la transition d'état (approximation pour le SoC)
def jacobian_state_transition():
    return np.array([[1]])


# Jacobienne de la mesure
def jacobian_measurement_function():
    return np.array([[1]])


# Boucle de simulation
num_steps = len(voltage_data)
print(f"Nombre total d'étapes : {num_steps}")
SoC_values = []

for t in range(num_steps):
    current = current_data[t]
    measured_voltage = voltage_data[t]
    temperature = temp_data[t]

    # Prédiction
    SoC_pred = SoC_est - np.array([[current * dt / nominal_capacity]])
    F = jacobian_state_transition()
    P_pred = F @ P @ F.T + Q

    # Tension prédite (choix entre charge/décharge selon le courant)
    mode = "charge" if current < 0 else "discharge"
    voltage_pred = voltage_model(SoC_pred[0, 0], current, temperature, mode=mode)

    # Mise à jour (filtrage)
    H = jacobian_measurement_function()
    K = P_pred @ H.T @ np.linalg.inv(H @ P_pred @ H.T + R)
    SoC_est = SoC_pred + K @ np.array([[measured_voltage - voltage_pred]])
    P = (np.eye(len(P)) - K @ H) @ P_pred

    # Stocker la valeur estimée de SoC
    SoC_values.append(SoC_est[0, 0])

# Affichage des résultats
plt.plot(range(num_steps), SoC_values, label="Estimation SoC")
plt.xlabel("Temps (s)")
plt.ylabel("État de Charge (SoC)")
plt.title("Estimation du SoC avec un Filtre de Kalman Étendu")
plt.legend()
plt.show()
