"""
calculate_energy_dissipation.py

Usage: python calculate_energy_dissipation.py

This script calculates the deterministic average energy dissipated and the random-average energy dissipated for a full-adder circuit at a particular drain voltage.
Probabilities between 0 and 1 in steps of 0.1 are used for all Cin, A, and B for a total of 1331 probability combinations
It also plots the percentage difference between the two quantities for each combination of pA and pB per figure, for a total of 11 figures
"""


import matplotlib.pyplot as plt
import numpy as np
import csv, time, os

pC = np.linspace(0.0, 1.0, 11)
pA = np.linspace(0.0, 1.0, 11)
pB = np.linspace(0.0, 1.0, 11)
Q_tau = np.zeros(64)
T = np.zeros(64)
percentage_difference = np.zeros((11, 11, 11))

def main():
    start = time.time_ns()

    if os.path.exists(f"./V_D-5.0/energy_dissipation.csv"):
        os.remove(f"./V_D-5.0/energy_dissipation.csv")

    with open("./V_D-5.0/energy_dissipation.csv", "a") as file:
        writer = csv.DictWriter(file, fieldnames=["pC", "pA", "pB", "Q_stochastic", "Q_input_state", "Percentage Difference"], lineterminator="\n")
        writer.writeheader()

    # Get the energy matrix at propagation delay time
    with open("./V_D-5.0/Summary.csv", "r") as file:
        reader = csv.DictReader(file)
        i = 0
        for row in reader:
            Q_tau[i] = float(row["Energy Dissipation at tau (kT)"])
            i += 1

    # Loop across all C probabilities
    for i in range(11):
        # Loop across all A probabilities
        for j in range(11):
            # Loop across all B probabilities
            for k in range(11):
                # Fill in inputs for all six input variables
                before_C = np.random.choice([0, 1], p=[1-pC[i], pC[i]], size=1000)
                before_A = np.random.choice([0, 1], p=[1-pA[j], pA[j]], size=1000)
                before_B = np.random.choice([0, 1], p=[1-pB[k], pB[k]], size=1000)
                after_C = np.random.choice([0, 1], p=[1-pC[i], pC[i]], size=1000)
                after_A = np.random.choice([0, 1], p=[1-pA[j], pA[j]], size=1000)
                after_B = np.random.choice([0, 1], p=[1-pB[k], pB[k]], size=1000)

                index = before_C*32 + before_A*16 + before_B*8 + after_C*4 + after_A*2 + after_B*1
                Q = 0

                # Loop across all trials
                for l in index:
                    Q += Q_tau[l]
                Q /= 1000

                for a in range(2):
                    if a == 0:
                        p_a = 1-pC[i]
                    else:
                        p_a = pC[i]
                    for b in range(2):
                        if b == 0:
                            p_b = 1-pA[j]
                        else:
                            p_b = pA[j]
                        for c in range(2):
                            if c == 0:
                                p_c = 1-pB[k]
                            else:
                                p_c = pB[k]
                            for d in range(2):
                                if d == 0:
                                    p_d = 1-pC[i]
                                else:
                                    p_d = pC[i]
                                for e in range(2):
                                    if e == 0:
                                        p_e = 1-pA[j]
                                    else:
                                        p_e = pA[j]
                                    for f in range(2):
                                        if f == 0:
                                            p_f = 1-pB[k]
                                        else:
                                            p_f = pB[k]

                                        index = a*32 + b*16 + c*8 +d*4 + e*2 + f*1
                                        T[index] = p_a*p_b*p_c*p_d*p_e*p_f

                W = np.multiply(Q_tau, T)
                W = np.sum(W)

                diff = ((Q-W)/Q)*100

                with open("./V_D-5.0/energy_dissipation.csv", "a") as file:
                    writer = csv.DictWriter(file, fieldnames=["pC", "pA", "pB", "Q_stochastic", "Q_input_state", "Percentage Difference"], lineterminator="\n")
                    writer.writerow({"pC":pC[i], "pA":pA[j], "pB":pB[k], "Q_stochastic":Q, "Q_input_state":W, "Percentage Difference":diff})

    with open("./V_D-5.0/energy_dissipation.csv", "r") as file:
        reader = csv.DictReader(file)
        i = 0
        j = 0
        k = 0
        for row in reader:
            percentage_difference[i][j][k] = row["Percentage Difference"]
            k += 1

            if (k % 11 == 0) and (k > 0):
                k = 0
                j += 1

            if (j % 11 == 0) and (j > 0):
                j = 0
                i += 1

    for i in range(11):
        plot = plt.pcolormesh(pB, pA, percentage_difference[i])
        plt.xlabel("Probability of 1 in B")
        plt.ylabel("Probability of 1 in A")
        plt.title(f"Percentage difference for $p^C_1$ = {pC[i]:.1f}")
        plt.colorbar(plot)
        plt.savefig(f"./V_D-5.0/Percentage_Difference_Contour_pC-{pC[i]:.1f}.png", bbox_inches='tight')
        plt.close()
                      
    end = time.time_ns()

    elapsed = end - start

    print(f"Time elapsed is {elapsed/1e9:.2f} seconds")

if __name__ == "__main__":
    main()