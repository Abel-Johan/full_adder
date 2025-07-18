import matplotlib.pyplot as plt
import numpy as np
import csv, time, sys
from pathlib import Path
from full_adder_deterministic import convert_to_binary

# Define simulation parameters
tint = 100                          # Time interval between data points; normalised by beta*h_bar, dimensionless
T = 5000000                         # Total simulation time; normalised by beta*h_bar, dimensionless
Ntot = int(T/tint)                  # Number of data points
N_INPUTS = 8                        # Number of possible input combinations
ksi_th = 0.01                       # Error rate threshold

# Initialise x and y values
timesteps = np.arange(0, T, tint)   # x-axis: time
Sum = np.zeros(Ntot)                # Initialise sum output voltage vector
Cout = np.zeros(Ntot)               # Initialise carry out output voltage vector
ErrorSum = np.zeros(Ntot)           # Initialise sum error rate vector
ErrorCout = np.zeros(Ntot)          # Initialise carry out error rate vector
Qdiss = np.zeros(Ntot)              # Initialise total energy dissipation vector

# Initialise array to store propagation times
tau_sum = np.zeros((N_INPUTS, N_INPUTS))
tau_cout = np.zeros((N_INPUTS, N_INPUTS))

def plot_individual():
    try:
        Path("./Sum").mkdir()
    except FileExistsError:
        print("Directory already exists - no need to create again")
    try:
        Path("./CarryOut").mkdir()
    except FileExistsError:
        print("Directory already exists - no need to create again")
    try:
        Path("./ErrorSum").mkdir()
    except FileExistsError:
        print("Directory already exists - no need to create again")
    try:    
        Path("./ErrorCout").mkdir()
    except FileExistsError:
        print("Directory already exists - no need to create again")
    try:    
        Path("./Qdiss").mkdir()
    except FileExistsError:
        print("Directory already exists - no need to create again")

    for i in range(N_INPUTS):
        i_bin = convert_to_binary(i)
        for j in range(N_INPUTS):
            j_bin = convert_to_binary(j)
            with open(f"./Results/Results-Prev{i_bin}-Curr{j_bin}.csv", "r") as file:
                reader = csv.DictReader(file)
                k = 0
                for row in reader:
                    Sum[k] = row["Sum Voltage (V)"]
                    Cout[k] = row["Carry Out Voltage (V)"]
                    ErrorSum[k] = row["Sum Error Rate (Dimless)"]
                    ErrorCout[k] = row["Carry Out Error Rate (Dimless)"]
                    Qdiss[k] = row["Energy Dissipation (J)"]

                    # Sometimes, the concept of propagation time for a particular trial is trivial
                    # This is when the old theoretical steady-state state is equal to the new theoretical steady-state so error rate is always < 0.01
                    # This means tau would be zero for that trial -> the assignment of tau is quite redundant
                    # This first inequality ensures that the concept of tau is non-trivial
                    if float(ErrorSum[0]) > 0.01:
                        if float(ErrorSum[k]) < 0.01 and tau_sum[i, j] == 0.0:
                            tau_sum[i, j] = k
                    if float(ErrorCout[0]) > 0.01:
                        if float(ErrorCout[k]) < 0.01 and tau_cout[i, j] == 0.0:
                            tau_cout[i, j] = k                   

                    k += 1
            plt.plot(timesteps, Sum)
            plt.xlabel("Time (s)")
            plt.ylabel("Sum Voltage (V)")
            plt.title(f"Plot of Sum Voltage against Time for Current Input {j_bin}, Previous Input {i_bin}")
            plt.plot(tau_sum[i, j]*tint, Sum[int(tau_sum[i, j])], 'rx')
            plt.text(tau_sum[i, j]*tint, Sum[int(tau_sum[i, j])]*0.9, f"({tau_sum[i, j]*tint}, {Sum[int(tau_sum[i, j])]})")
            plt.savefig(f"./Sum/Sum-Prev{i_bin}-Curr{j_bin}")
            plt.close()

            plt.plot(timesteps, Cout)
            plt.xlabel("Time (s)")
            plt.ylabel("Carry Out Voltage (V)")
            plt.title(f"Plot of Carry Out Voltage against Time for Current Input {j_bin}, Previous Input {i_bin}")
            plt.plot(tau_cout[i, j]*tint, Cout[int(tau_cout[i, j])], 'rx')
            plt.text(tau_cout[i, j]*tint, Cout[int(tau_cout[i, j])]*0.9, f"({tau_cout[i, j]*tint}, {Cout[int(tau_cout[i, j])]})")
            plt.savefig(f"./CarryOut/Cout-Prev{i_bin}-Curr{j_bin}")
            plt.close()

            plt.plot(timesteps, ErrorSum)
            plt.xlabel("Time (s)")
            plt.ylabel("Sum Error Rate (Dimless)")
            plt.title(f"Plot of Sum Error Rate against Time for Current Input {j_bin}, Previous Input {i_bin}")
            plt.plot(tau_sum[i, j]*tint, ErrorSum[int(tau_sum[i, j])], 'rx')
            plt.text(tau_sum[i, j]*tint, ErrorSum[int(tau_sum[i, j])]*0.9, f"({tau_sum[i, j]*tint}, {ErrorSum[int(tau_sum[i, j])]})")
            plt.savefig(f"./ErrorSum/ErrorSum-Prev{i_bin}-Curr{j_bin}")
            plt.close()

            plt.plot(timesteps, ErrorCout)
            plt.xlabel("Time (s)")
            plt.ylabel("Carry Out Error Rate (Dimless)")
            plt.title(f"Plot of Carry Out Error Rate against Time for Current Input {j_bin}, Previous Input {i_bin}")
            plt.plot(tau_cout[i, j]*tint, ErrorCout[int(tau_cout[i, j])], 'rx')
            plt.text(tau_cout[i, j]*tint, ErrorCout[int(tau_cout[i, j])]*0.9, f"({tau_cout[i, j]*tint}, {ErrorCout[int(tau_cout[i, j])]})")
            plt.savefig(f"./ErrorCout/ErrorCout-Prev{i_bin}-Curr{j_bin}")
            plt.close()

            plt.plot(timesteps, Qdiss)
            plt.xlabel("Time (s)")
            plt.ylabel("Energy Dissipation (J)")
            plt.title(f"Plot of Cumulative Energy Dissipation against Time for Current Input {j_bin}, Previous Input {i_bin}")
            plt.savefig(f"./Qdiss/Qdiss-Prev{i_bin}-Curr{j_bin}")
            plt.close()

def plot_concise():
    try:
        Path("./SumConcise").mkdir()
    except FileExistsError:
        print("Directory already exists - no need to create again")
    try:
        Path("./CarryOutConcise").mkdir()
    except FileExistsError:
        print("Directory already exists - no need to create again")
    try:
        Path("./ErrorSumConcise").mkdir()
    except FileExistsError:
        print("Directory already exists - no need to create again")
    try:    
        Path("./ErrorCoutConcise").mkdir()
    except FileExistsError:
        print("Directory already exists - no need to create again")
    try:    
        Path("./QdissConcise").mkdir()
    except FileExistsError:
        print("Directory already exists - no need to create again")
    for i in range(N_INPUTS):
        i_bin = convert_to_binary(i)

        # Create figures and axes to plot each quantity on
        sumfig, sumax = plt.subplots()
        sumax.set_xlabel("Time (s)")
        sumax.set_ylabel("Sum Voltage (V)")
        sumax.set_title("Plot of Sum Voltage against Time")

        coutfig, coutax = plt.subplots()            
        coutax.set_xlabel("Time (s)")
        coutax.set_xlabel("Carry Out Voltage (V)")
        coutax.set_title("Plot of Carry Out Voltage against Time")

        sumerrorfig, sumerrorax = plt.subplots()           
        sumerrorax.set_xlabel("Time (s)")
        sumerrorax.set_ylabel("Sum Error Rate (Dimless)")
        sumerrorax.set_title("Plot of Sum Error Rate against Time")

        couterrorfig, couterrorax = plt.subplots()            
        couterrorax.set_xlabel("Time (s)")
        couterrorax.set_label("Carry Out Error Rate (Dimless)")
        couterrorax.set_title("Plot of Carry Out Error Rate against Time")

        energyfig, energyax = plt.subplots()            
        energyax.set_xlabel("Time (s)")
        energyax.set_ylabel("Energy Dissipation (J)")
        energyax.set_title(f"Plot of Cumulative Energy Dissipation against Time")
        for j in range(N_INPUTS):
            j_bin = convert_to_binary(j)
            with open(f"./Results/Results-Prev{i_bin}-Curr{j_bin}.csv", "r") as file:
                reader = csv.DictReader(file)
                k = 0
                for row in reader:
                    Sum[k] = row["Sum Voltage (V)"]
                    Cout[k] = row["Carry Out Voltage (V)"]
                    ErrorSum[k] = row["Sum Error Rate (Dimless)"]
                    ErrorCout[k] = row["Carry Out Error Rate (Dimless)"]
                    Qdiss[k] = row["Energy Dissipation (J)"]
                    
                    # Sometimes, the concept of propagation time for a particular trial is trivial
                    # This is when the old theoretical steady-state state is equal to the new theoretical steady-state so error rate is always < 0.01
                    # This means tau would be zero for that trial -> the assignment of tau is quite redundant
                    # This first inequality ensures that the concept of tau is non-trivial
                    if float(ErrorSum[0]) > 0.01:
                        if float(ErrorSum[k]) < 0.01 and tau_sum[i, j] == 0.0:
                            tau_sum[i, j] = k
                    if float(ErrorCout[0]) > 0.01:
                        if float(ErrorCout[k]) < 0.01 and tau_cout[i, j] == 0.0:
                            tau_cout[i, j] = k                   
                            
                    k += 1

            #plt.plot(tau_cout[i, j]*tint, ErrorCout[int(tau_cout[i, j])], 'rx')
            sumax.plot(timesteps, Sum)
            sumax.plot(tau_sum[i, j]*tint, Sum[int(tau_sum[i, j])], 'gx')
            sumax.text(tau_sum[i, j]*tint, Sum[int(tau_sum[i, j])]*0.9, f"({tau_sum[i, j]*tint}, {Sum[int(tau_sum[i, j])]})")
            coutax.plot(timesteps, Cout)
            coutax.plot(tau_cout[i, j]*tint, Cout[int(tau_cout[i, j])], 'gx')
            coutax.text(tau_cout[i, j]*tint, Cout[int(tau_cout[i, j])]*0.9, f"({tau_cout[i, j]*tint}, {Cout[int(tau_cout[i, j])]})")
            sumerrorax.plot(timesteps, ErrorSum)
            sumerrorax.plot(tau_sum[i, j]*tint, ErrorSum[int(tau_sum[i, j])], 'gx')
            sumerrorax.text(tau_sum[i, j]*tint, ErrorSum[int(tau_sum[i, j])]*0.9, f"({tau_sum[i, j]*tint}, {ErrorSum[int(tau_sum[i, j])]})")
            couterrorax.plot(timesteps, ErrorCout)
            couterrorax.plot(tau_cout[i, j]*tint, ErrorCout[int(tau_cout[i, j])], 'gx')
            couterrorax.text(tau_cout[i, j]*tint, ErrorCout[int(tau_sum[i, j])]*0.9, f"({tau_cout[i, j]*tint}, {ErrorCout[int(tau_cout[i, j])]})")
            energyax.plot(timesteps, Qdiss)
           
        sumfig.savefig(f"./SumConcise/Sum-Concise-Prev{i_bin}")
        coutfig.savefig(f"./CarryOutConcise/Cout-Concise-Prev{i_bin}")
        sumerrorfig.savefig(f"./ErrorSumConcise/ErrorSum-Concise-Cout{i_bin}")
        couterrorfig.savefig(f"./ErrorCoutConcise/ErrorCout-Concise-Prev{i_bin}")
        energyfig.savefig(f"./QdissConcise/Qdiss-Concise-Prev{i_bin}")
        plt.close()

def main():
    start = time.time_ns()

    if len(sys.argv) != 2:
        print("Usage: python plot_full_adder.py [individual|concise]")
        sys.exit(1)

    if sys.argv[1] == "individual":
        plot_individual()
    elif sys.argv[1] == "concise":
        plot_concise()

    end = time.time_ns()

    elapsed = end - start

    print(f"Time elapsed is {elapsed/1e9:.2f} seconds")

if __name__ == "__main__":
    main()