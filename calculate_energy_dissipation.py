import matplotlib.pyplot as plt
import numpy as np
import csv, time

pC = np.linspace(0.0, 1.0, 11)
pA = np.linspace(0.0, 1.0, 11)
pB = np.linspace(0.0, 1.0, 11)
Q_tau = np.zeros(64)

def main():
    start = time.time_ns()

    # Get the energy matrix at propagation delay time
    with open("V_D-5.0/Summary.csv", "r") as file:
        reader = csv.DictReader(file, fieldnames=["Previous Input" ,"Current Input", "Sum Propagation Delay", "Cout Propagation Delay", "Energy Dissipation at tau (kT)", "Energy Dissipation Total (kT)"])
        i = 0
        for row in reader:
            Q_tau[i] = row["Energy Dissipation at tau (kT)"]
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

    end = time.time_ns()

    elapsed = end - start

    print(f"Time elapsed is {elapsed/1e9:.2f} seconds")

if __name__ == "__main__":
    main()