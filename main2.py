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


test_str = str(input("Run test? (Y/n) ")).lower()
TEST = True if test_str == "y" else False
print(f"Running in {"test mode" if TEST else "prod mode"}")


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
    print("Reading ./test.csv file...")
    data = pd.read_csv("test.csv", sep=';', header=None) # ? sep correct ?
    #data = pd.read_csv("test.csv", header=None) # ?

data.columns = ['voltage', 'current_inv', 'SOC_true', 'temperature']
print(data.head()) # view the first few lines of the dataframe to manually verify the data has been read correctly
#input("Run algorithm? Press Enter to continue...")

# Extraction des colonnes par index
voltage_data = data['voltage'].values
current_data = data['current_inv'].values
soc_true_data = data['SOC_true'].values
temperature_data = data['temperature'].values

# Paramètres de la batterie
nominal_capacity = 280.0  # Capacité nominale en Ah (ajuster en fonction de la batterie)
dt = 1.0  # Pas de temps en secondes
initial_voltage = voltage_data[0]

initial_SoC = soc_true_data[0]
#initial_SoC = np.interp(initial_voltage, data_ocv_discharge["data_U"], data_ocv_discharge["data_SOC"])


# Kalman filter initialization
SoC_est = np.array([[initial_SoC]])
P = np.array([[1]])     # tune this value
Q = np.array([[1e-10]]) # tune this value
R = np.array([[0.05]])  # tune this value


# Modèle de tension utilisant les courbes de charge et décharge
def voltage_model(SOC, current, temperature=None, mode="discharge"): # temp not used YET
    if mode == "charge":
        soc_ocv = data_ocv_charge["data_SOC"].values
        voltage_ocv = data_ocv_charge["data_U"].values
        R_int = 0.06 #R_int_chg
        # todo: use hppc test data to get R_int for current SoC
        # constant R value is sufficiently accurate for now
    else:
        soc_ocv = data_ocv_discharge["data_SOC"].values
        voltage_ocv = data_ocv_discharge["data_U"].values
        R_int = -0.01 #R_int_dsg
        # todo: use hppc test data to get R_int for current SoC
        # constant R value is sufficiently accurate for now

    # Mise à l'échelle de SOC si nécessaire
    SOC_scaled = SOC  # Si les courbes utilisent une échelle de 0 à 1
    V_oc = np.interp(SOC_scaled, soc_ocv, voltage_ocv)
    return V_oc - current * R_int

# Jacobienne de la transition d'état (approximation pour le SoC)
def jacobian_state_transition():
    return np.array([[1]])

## Try this one out :
#def jacobian_state_transition(current, dt, nominal_capacity):
#    # For simplicity, assuming a linear model with respect to current
#    # but you could include a term for temperature or resistance changes
#    jacobian = np.array([[1 - (current * dt / nominal_capacity)]])
#    return jacobian

# Jacobienne de la mesure
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

# Initialisation pour le calcul de l'erreur maximale
max_ae = 0

# Boucle de simulation
num_steps = len(voltage_data)
print(f"Total nb of steps : {num_steps}")
SoC_values = []
#error = []
OffSet = soc_true_data[0]
OffSet = 0
factor = -35.341 # -30.82 # -35.341

for t in range(num_steps):
    current = current_data[t]/(-35.341) # -30.82 # -35.341
    #current = np.mean(current_data[:5]) / -30.82
    measured_voltage = voltage_data[t]
    temperature = temperature_data[t]

    # Prédiction
    SoC_pred = SoC_est - np.array([[current * dt / nominal_capacity]])
    F = jacobian_state_transition()
    P_pred = F @ P @ F.T + Q

    # Tension prédite (choix entre charge/décharge selon le courant)
    mode = "charge" if current > 0 else "discharge"
    voltage_pred = voltage_model(SoC_pred[0, 0], current, mode=mode)

    # Mise à jour (filtrage)
    H = jacobian_measurement_function(SoC_pred[0, 0], current)
    K = P_pred @ H.T @ np.linalg.inv(H @ P_pred @ H.T + R + 1e-9 * np.eye(H.shape[0]))
    innovation = measured_voltage - voltage_pred
    SoC_est = SoC_pred + K @ np.array([[measured_voltage - voltage_pred]])
    P = (np.eye(len(P)) - K @ H) @ P_pred

    # Diviser SoC_est par 100 avant de stocker et calculer l'erreur
    #SoC_est /= 1.4

    # Stocker la valeur estimée de SoC
    SoC_values.append(SoC_est[0, 0])

    # Calcul de l'erreur absolue maximale
    actual_soc = soc_true_data[t]
    #error.append(abs(actual_soc - OffSet - SoC_est[0, 0]) /SoC_est[0,0])

    absolute_error = abs(SoC_est[0, 0] - actual_soc)
    if absolute_error > max_ae:
        max_ae = absolute_error
    #max_ae = max(max_ae, abs(SoC_est[0, 0] - actual_soc))



# Affichage des résultats

print('max soc estim', max(SoC_values))
print('max vrai soc', max(soc_true_data))

#SoC_values = [((x+soc_true_data[0])/max(SoC_values))*100 for x in SoC_values]
#soc_true_data = [x for x in soc_true_data]


# Affichage des résultats
#max_SoC_values = max(SoC_values)
#SoC_values += initial_SoC  # Ajouter le SoC initial
#SoC_values /= max_SoC_values  # Normaliser
#SoC_values *= 100  # Convertir en pourcentage
#SoC_values *= -1  # Inverser pour correspondre aux données réelles
#SoC_values += initial_SoC  # Ajouter le SoC initial


#SoC_values = [((x+soc_true_data[0])/max(SoC_values))*100 for x in SoC_values]
#soc_true_data = [x for x in soc_true_data]

#print('max soc estim', max(SoC_values))
#print('max vrai soc', max(soc_true_data))

# Fonction pour créer un fichier CSV à partir d'une liste de données
#def creer_fichier_csv(nom_fichier, donnees):
    #with open(nom_fichier, mode='w', newline='') as fichier:
        #writer = csv.writer(fichier)
        #writer.writerows(donnees)
    #print(f"Fichier '{nom_fichier}' créé avec succès.")

# Création des deux fichiers CSV
#creer_fichier_csv('actual_soc.csv', actual_soc)
#creer_fichier_csv('fichier2.csv', soc_true_data)


# Affichage de l'erreur maximale
print(f"Maximum Absolute Error (MaxAE) : {max_ae} %")

stop = timeit.default_timer()

print('Time: ', stop - start, 's')

SoC_values = np.array(SoC_values) #+ OffSet

#error = abs(statistics.mean(error))
#print("error", abs(1-error)*100, ' %')


squared_errors = [(actual_soc + OffSet - est_soc)**2 for actual_soc, est_soc in zip(soc_true_data, SoC_values)] # list of squared errors # yo the zip() function is actually so cool so useful
mse = sum(squared_errors) / len(squared_errors)  # Mean Squared Error
rmse = np.sqrt(mse) # Root Mean Squared Error
print(f"Root Mean Square Error (RMSE): {rmse} %")


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