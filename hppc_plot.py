import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#choice = int(input("1. Charge, 2. Discharge: "))
choice = 2 # 1 or 2

if choice == 1:
    print("Getting data from HPPC files...")
    data_HCCP = pd.read_excel("data/EVE_HPPC_1_25degree_CHG-injectionTemplate.xlsx", header=3, usecols="A:D")
    data_HCCP.columns = ['steps', 'voltage', 'current_inv', 'SOC_true']
    print("data_hppc_chg (head):")
    print(data_HCCP.head())
elif choice == 2:
    data_HCCP = pd.read_excel("data/EVE_HPPC_1_25degree_DSG-injectionTemplate.xlsx", header=3, usecols="A:D")
    data_HCCP.columns = ['steps', 'voltage', 'current_inv', 'SOC_true']
    print("data_hppc_dsg (head):")
    print(data_HCCP.head())
else:
    print("Invalid choice")
    exit(1)

step_data = data_HCCP['steps'].values
voltage_data = data_HCCP['voltage'].values
current_data = data_HCCP['current_inv'].values
soc_true_data = data_HCCP['SOC_true'].values


# Fonction pour calculer l'impédance
def calculate_impedance(voltage_pulse, voltage_rest, current):
    """
    Calcule l'impédance à partir des données de tension et de courant.
    voltage_pulse : Tension pendant l'impulsion (V)
    voltage_rest  : Tension en régime stationnaire (V)
    current       : Courant appliqué (A)

    Retourne l'impédance (Ohms)
    """
    return (voltage_pulse - voltage_rest) / current


# Exemple de données d'entrée
steps = len(step_data)  # Nombre de steps dans le test HPPC
voltages_pulse = voltage_data  # Tension pendant l'impulsion (V)
voltages_rest = [0 for x in range(steps)]  # Tension en régime stationnaire (V)
currents = current_data  # Courant appliqué pendant l'impulsion (A)
soc_values = soc_true_data  # Valeurs d'état de charge (SOC) en % (exemple)

# Calcul de l'impédance pour chaque step
impedances = np.array([calculate_impedance(voltages_pulse[i], voltages_rest[i], currents[i]) for i in range(steps)])

# Affichage des résultats
print("SOC (%) | Impedance (Ohms)")
for i in range(steps):
    print(f"{soc_values[i]:3}  % | {impedances[i]:.4f}  Ohms")

# Visualisation de l'impédance en fonction du SOC
plt.plot(soc_values, impedances, marker='o')
plt.title("Impedance variation as a function of SOC")
plt.xlabel("State Of Charge (SOC) [%]")
plt.ylabel("Impedance [Ohms]")
plt.grid(True)
plt.show()

# TEST:
test_SoC = 50
resistance = np.interp(test_SoC, soc_values, impedances)
print(f"Interpolated resistance at {test_SoC}% SOC: {resistance:.4f} Ohms")
