"""
plot_full_adder_deterministic.py

Usage: python plot_full_adder_deterministic.py
OR     python plot_full_adder_deterministic.py individual
OR     python plot_full_adder_deterministic.py concise

This script plots time-evolution graphs of Sum, Carry Out, Sum Error Rate, Carry Out Error Rate, and Cumulative Energy Dissipation
'individual' plots one graph per figure. This option also creates a Summary.csv file which gives the propagation delay, cumulative energy dissipated by tau, and total energy dissipated for each combination of previous and current input
'concise' plots all 8 possible current inputs for a given previous input on the same figure
Default is do both options

ALL VARIABLES AND QUANTITIES USED ARE DIMENSIONLESS, UNLESS OTHERWISE STATED

The following quantities are not used in the code, but are required to understand the physics of the model:
beta  = 1/(kB*T)        # Coldness. kB = Boltzmann constant, T = absolute temperature
h_bar = h/(2*pi)        # Reduced Planck's constant. h = Planck's constant
q                       # Elementary charge, roughly equal to 1.602e-19 C
V_T   = (kB*T)/q        # Thermal voltage
"""


import matplotlib.pyplot as plt
import numpy as np
import csv, time, sys, os
from pathlib import Path
from full_adder_deterministic import convert_to_binary

# Define simulation parameters
tint = 50000                          # Time interval between data points, normalised by beta*h_bar
T = 1000000000                         # Total simulation time, normalised by beta*h_bar
Ntot = int(T/tint)                  # Number of data points
N_INPUTS = 8                        # Number of possible input combinations
ksi_th = 0.01                       # Error rate threshold
V_D = 12.5                           # Drain voltage, normalised by V_T
kT = 4.143e-21                      # What we normalised energy with. Used to rescale energy dissipation to prevent overflow

# Initialise x and y values
timesteps = np.arange(0, T, tint)   # x-axis: time
marker_timesteps = np.linspace(0, T, 11) # For plotting markers
Sum = np.zeros(Ntot)                # Initialise sum output voltage vector
Cout = np.zeros(Ntot)               # Initialise carry out output voltage vector
ErrorSum = np.zeros(Ntot)           # Initialise sum error rate vector
ErrorCout = np.zeros(Ntot)          # Initialise carry out error rate vector
Qdiss = np.zeros(Ntot)              # Initialise total energy dissipation vector

# Initialise array to store propagation times
tau_sum = np.zeros((N_INPUTS, N_INPUTS))
tau_cout = np.zeros((N_INPUTS, N_INPUTS))

def round_to_3(x):
    return round(x, -int(np.floor(np.log10(abs(x))))+2)

def plot_individual():
    if os.path.exists(f"./V_D-{V_D}/Summary.csv"):
        os.remove(f"./V_D-{V_D}/Summary.csv")

    with open(f"./V_D-{V_D}/Summary.csv", "a") as file:
        writer = csv.DictWriter(file, fieldnames=["Previous Input", "Current Input", "Sum Propagation Delay", "Cout Propagation Delay", "Energy Dissipation at tau (kT)", "Energy Dissipation Total (kT)"], lineterminator="\n")
        writer.writeheader()

    sum_dir = f"./V_D-{V_D}/Sum-{V_D}"
    cout_dir = f"./V_D-{V_D}/CarryOut-{V_D}"
    errorsum_dir = f"./V_D-{V_D}/ErrorSum-{V_D}"
    errorcout_dir = f"./V_D-{V_D}/ErrorCout-{V_D}"
    qdiss_dir = f"./V_D-{V_D}/Qdiss-{V_D}"
    try:
        Path(sum_dir).mkdir()
    except FileExistsError:
        print("Directory already exists - no need to create again")
    try:
        Path(cout_dir).mkdir()
    except FileExistsError:
        print("Directory already exists - no need to create again")
    try:
        Path(errorsum_dir).mkdir()
    except FileExistsError:
        print("Directory already exists - no need to create again")
    try:    
        Path(errorcout_dir).mkdir()
    except FileExistsError:
        print("Directory already exists - no need to create again")
    try:    
        Path(qdiss_dir).mkdir()
    except FileExistsError:
        print("Directory already exists - no need to create again")

    for i in range(N_INPUTS):
        i_bin = convert_to_binary(i)
        for j in range(N_INPUTS):
            j_bin = convert_to_binary(j)
            with open(f"./V_D-{V_D}/ResultsV_D-{V_D}/Results-Prev{i_bin}-Curr{j_bin}.csv", "r") as file:
                reader = csv.DictReader(file)
                k = 0
                for row in reader:
                    Sum[k] = row["Sum Voltage (V)"]
                    Cout[k] = row["Carry Out Voltage (V)"]
                    ErrorSum[k] = row["Sum Error Rate (Dimless)"]
                    ErrorCout[k] = row["Carry Out Error Rate (Dimless)"]
                    Qdiss[k] = float(row["Energy Dissipation (J)"])/kT

                    k += 1

            maxerrorsum = np.max(ErrorSum)
            maxerrorsum_index = np.argmax(ErrorSum)
            maxerrorcout = np.max(ErrorCout)
            maxerrorcout_index = np.argmax(ErrorCout)

            if maxerrorsum > ksi_th:
                for k in range(maxerrorsum_index, Ntot):
                    if float(ErrorSum[k]) < ksi_th:
                        tau_sum[i, j] = k
                        break
                    elif k == Ntot - 1:
                        tau_sum[i, j] = k
            
            if maxerrorcout > ksi_th:
                for k in range(maxerrorcout_index, Ntot):
                    if float(ErrorCout[k]) < ksi_th:
                        tau_cout[i, j] = k
                        break
                    elif k == Ntot - 1:
                        tau_cout[i, j] = k
                
            plt.plot(timesteps, Sum)
            plt.rcParams["figure.figsize"] = (7.2, 4.8)
            plt.xlabel("Time ($βℏ$)")
            plt.ylabel("Sum Voltage ($V_T$)")
            plt.title(f"Plot of Sum Voltage against Time for Current Input {j_bin}, Previous Input {i_bin}")
            plt.plot(tau_sum[i, j]*tint, Sum[int(tau_sum[i, j])], 'rx')
            plt.text(tau_sum[i, j]*tint, Sum[int(tau_sum[i, j])]*0.9, f"({tau_sum[i, j]*tint}, {Sum[int(tau_sum[i, j])]})")
            plt.savefig(f"{sum_dir}/Sum-Prev{i_bin}-Curr{j_bin}")
            plt.close()

            plt.plot(timesteps, Cout)
            plt.rcParams["figure.figsize"] = (7.6, 4.8)
            plt.xlabel("Time ($βℏ$)")
            plt.ylabel("Carry Out Voltage ($V_T$)")
            plt.title(f"Plot of Carry Out Voltage against Time for Current Input {j_bin}, Previous Input {i_bin}")
            plt.plot(tau_cout[i, j]*tint, Cout[int(tau_cout[i, j])], 'rx')
            plt.text(tau_cout[i, j]*tint, Cout[int(tau_cout[i, j])]*0.9, f"({tau_cout[i, j]*tint}, {Cout[int(tau_cout[i, j])]})")
            plt.savefig(f"{cout_dir}/Cout-Prev{i_bin}-Curr{j_bin}")
            plt.close()

            plt.plot(timesteps, ErrorSum)
            plt.rcParams["figure.figsize"] = (7.2, 4.8)
            plt.xlabel("Time ($βℏ$)")
            plt.ylabel("Sum Error Rate (Dimless)")
            plt.title(f"Plot of Sum Error Rate against Time for Current Input {j_bin}, Previous Input {i_bin}")
            plt.plot(tau_sum[i, j]*tint, ErrorSum[int(tau_sum[i, j])], 'rx')
            plt.text(tau_sum[i, j]*tint, ErrorSum[int(tau_sum[i, j])]*0.9, f"({tau_sum[i, j]*tint}, {ErrorSum[int(tau_sum[i, j])]})")
            plt.savefig(f"{errorsum_dir}/ErrorSum-Prev{i_bin}-Curr{j_bin}")
            plt.close()

            plt.plot(timesteps, ErrorCout)
            plt.rcParams["figure.figsize"] = (7.6, 4.8)
            plt.xlabel("Time ($βℏ$)")
            plt.ylabel("Carry Out Error Rate (Dimless)")
            plt.title(f"Plot of Carry Out Error Rate against Time for Current Input {j_bin}, Previous Input {i_bin}")
            plt.plot(tau_cout[i, j]*tint, ErrorCout[int(tau_cout[i, j])], 'rx')
            plt.text(tau_cout[i, j]*tint, ErrorCout[int(tau_cout[i, j])]*0.9, f"({tau_cout[i, j]*tint}, {ErrorCout[int(tau_cout[i, j])]})")
            plt.savefig(f"{errorcout_dir}/ErrorCout-Prev{i_bin}-Curr{j_bin}")
            plt.close()

            plt.plot(timesteps, Qdiss)
            plt.rcParams["figure.figsize"] = (12.0, 4.8)
            plt.xlabel("Time ($βℏ$)")
            plt.ylabel("Energy Dissipation ($kT$)")
            plt.title(f"Plot of Cumulative Energy Dissipation against Time for Current Input {j_bin}, Previous Input {i_bin}")
            plt.savefig(f"{qdiss_dir}/Qdiss-Prev{i_bin}-Curr{j_bin}")
            plt.close()

            with open(f"./V_D-{V_D}/Summary.csv", "a") as file:
                writer = csv.DictWriter(file, fieldnames=["Previous Input", "Current Input", "Sum Propagation Delay (βℏ)", "Cout Propagation Delay (βℏ)", "Energy Dissipation at τ (kT)", "Energy Dissipation Total (kT)"], lineterminator="\n")
                writer.writerow({"Previous Input": i_bin, "Current Input": j_bin, "Sum Propagation Delay (βℏ)": tau_sum[i, j]*tint, "Cout Propagation Delay (βℏ)": tau_cout[i, j]*tint, "Energy Dissipation at τ (kT)": Qdiss[9683], "Energy Dissipation Total (kT)": Qdiss[-1]})
    #print(f"Propagation time = {np.max(np.concatenate((tau_sum, tau_cout)))*tint}")

def plot_concise():
    sum_dir = f"./V_D-{V_D}/SumConcise-{V_D}"
    cout_dir = f"./V_D-{V_D}/CarryOutConcise-{V_D}"
    errorsum_dir = f"./V_D-{V_D}/ErrorSumConcise-{V_D}"
    errorcout_dir = f"./V_D-{V_D}/ErrorCoutConcise-{V_D}"
    qdiss_dir = f"./V_D-{V_D}/QdissConcise-{V_D}"
    try:
        Path(sum_dir).mkdir()
    except FileExistsError:
        print("Directory already exists - no need to create again")
    try:
        Path(cout_dir).mkdir()
    except FileExistsError:
        print("Directory already exists - no need to create again")
    try:
        Path(errorsum_dir).mkdir()
    except FileExistsError:
        print("Directory already exists - no need to create again")
    try:    
        Path(errorcout_dir).mkdir()
    except FileExistsError:
        print("Directory already exists - no need to create again")
    try:    
        Path(qdiss_dir).mkdir()
    except FileExistsError:
        print("Directory already exists - no need to create again")
    for i in range(N_INPUTS):
        i_bin = convert_to_binary(i)

        # Create figures and axes to plot each quantity on
        sumfig, sumax = plt.subplots()
        sumax.set_xlabel("Time ($βℏ$)")
        sumax.set_ylabel("Sum Voltage ($V_T$)")
        sumax.set_title("Plot of Sum Voltage against Time")

        coutfig, coutax = plt.subplots()            
        coutax.set_xlabel("Time ($βℏ$)")
        coutax.set_xlabel("Carry Out Voltage ($V_T$)")
        coutax.set_title("Plot of Carry Out Voltage against Time")

        sumerrorfig, sumerrorax = plt.subplots()           
        sumerrorax.set_xlabel("Time ($βℏ$)")
        sumerrorax.set_ylabel("Sum Error Rate (Dimless)")
        sumerrorax.set_title("Plot of Sum Error Rate against Time")

        couterrorfig, couterrorax = plt.subplots()            
        couterrorax.set_xlabel("Time ($βℏ$)")
        couterrorax.set_label("Carry Out Error Rate (Dimless)")
        couterrorax.set_title("Plot of Carry Out Error Rate against Time")

        energyfig, energyax = plt.subplots()    
        energyfig.set_size_inches(7.2, 4.8)        
        energyax.set_xlabel("Time ($βℏ$)")
        energyax.set_ylabel("Energy Dissipation ($kT$)")
        energyax.set_title(f"Plot of Cumulative Energy Dissipation against Time")

        markers = ['o', 'v', '^', '+', 'x', 's', '<', '>']
        colours = ['b', 'g', 'r', 'orange', 'm', 'y', 'k', 'brown']
        #x_multiplier = [1.02, 1.02, 1.02, 0.72, 1.02, 0.72, 0.72, 0.72]
        #y_multiplier = [0.42, 0.73, 0.62, 1.17, 0.53, 1.23, 1.32, 1.2]

        Q_max = 0
        for j in range(N_INPUTS):
            j_bin = convert_to_binary(j)
            with open(f"./V_D-{V_D}/ResultsV_D-{V_D}/Results-Prev{i_bin}-Curr{j_bin}.csv", "r") as file:
                reader = csv.DictReader(file)
                k = 0
                for row in reader:
                    Sum[k] = row["Sum Voltage (V)"]
                    Cout[k] = row["Carry Out Voltage (V)"]
                    ErrorSum[k] = row["Sum Error Rate (Dimless)"]
                    ErrorCout[k] = row["Carry Out Error Rate (Dimless)"]
                    Qdiss[k] = float(row["Energy Dissipation (J)"])/kT
                    
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

            sumax.plot(timesteps, Sum, markers[j], ls='-', c=colours[j], markevery=int(Ntot/10), label=j_bin)
            # sumax.plot(tau_sum[i, j]*tint, Sum[int(tau_sum[i, j])], 'gx')
            # sumax.text(tau_sum[i, j]*tint, Sum[int(tau_sum[i, j])]*0.9, f"({tau_sum[i, j]*tint}, {Sum[int(tau_sum[i, j])]})")
            coutax.plot(timesteps, Cout, markers[j], ls='-', c=colours[j], markevery=int(Ntot/10), label=j_bin)
            # coutax.plot(tau_cout[i, j]*tint, Cout[int(tau_cout[i, j])], 'gx')
            # coutax.text(tau_cout[i, j]*tint, Cout[int(tau_cout[i, j])]*0.9, f"({tau_cout[i, j]*tint}, {Cout[int(tau_cout[i, j])]})")
            sumerrorax.plot(timesteps, ErrorSum, markers[j], ls='-', c=colours[j], markevery=int(Ntot/10), label=j_bin)
            # sumerrorax.plot(tau_sum[i, j]*tint, ErrorSum[int(tau_sum[i, j])], 'gx')
            # sumerrorax.text(tau_sum[i, j]*tint, ErrorSum[int(tau_sum[i, j])]*0.9, f"({tau_sum[i, j]*tint}, {ErrorSum[int(tau_sum[i, j])]})")
            couterrorax.plot(timesteps, ErrorCout, markers[j], ls='-', c=colours[j], markevery=int(Ntot/10), label=j_bin)
            # couterrorax.plot(tau_cout[i, j]*tint, ErrorCout[int(tau_cout[i, j])], 'gx')
            # couterrorax.text(tau_cout[i, j]*tint, ErrorCout[int(tau_sum[i, j])]*0.9, f"({tau_cout[i, j]*tint}, {ErrorCout[int(tau_cout[i, j])]})")
            energyax.plot(timesteps, Qdiss, markers[j], ls='-', c=colours[j], markevery=int(Ntot/10), label=j_bin)
            if Qdiss[-1] > Q_max:
                Q_max = Qdiss[-1]
            #energyax.text(1531200*x_multiplier[j], Qdiss[15312]*y_multiplier[j], f"({round_to_3(Qdiss[15312])})", c=colours[j], backgroundcolor='w')
      
        sumax.legend(loc=7,fontsize=11)
        coutax.legend(loc=7,fontsize=11)
        sumerrorax.legend(loc=7,fontsize=11)
        couterrorax.legend(loc=7,fontsize=11)
        energyax.legend(loc=7,fontsize=11)

        energyax.vlines(484150000, 0, Q_max, 'r', 'dashed')
        energyax.text(484150000, 0, "$t=τ$", color='r')

        sumfig.savefig(f"{sum_dir}/Sum-Concise-Prev{i_bin}")
        coutfig.savefig(f"{cout_dir}/Cout-Concise-Prev{i_bin}")
        sumerrorfig.savefig(f"{errorsum_dir}/ErrorSum-Concise-Cout{i_bin}")
        couterrorfig.savefig(f"{errorcout_dir}/ErrorCout-Concise-Prev{i_bin}")
        energyfig.savefig(f"{qdiss_dir}/Qdiss-Concise-Prev{i_bin}")
        plt.close()

def main():
    start = time.time_ns()

    if len(sys.argv) == 2:
        if sys.argv[1] == "individual":
            plot_individual()
        elif sys.argv[1] == "concise":
            plot_concise()
    elif len(sys.argv) == 1:
        plot_individual()
        plot_concise()
    else:
        print("Usage: python plot_full_adder_deterministic.py [individual|concise] OR python plot_full_adder_deterministic.py")
        sys.exit(1)

    end = time.time_ns()

    elapsed = end - start

    print(f"Time elapsed is {elapsed/1e9:.2f} seconds")

if __name__ == "__main__":
    main()