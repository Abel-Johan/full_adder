"""
full_adder_gillespie.py

This script models the time-evolution of the output voltage of a full-adder circuit.
The full-adder circuit is to be constructed using only NANDs and NOTs.
It uses a random, probabilistic approach by ...
The model takes into account the changes in behaviour if the previous input to the adder was different

ALL VARIABLES AND QUANTITIES USED ARE DIMENSIONLESS, UNLESS OTHERWISE STATED

The following quantities are not used in the code, but are required to understand the physics of the model:
beta  = 1/(kB*T)        # Coldness. kB = Boltzmann constant, T = absolute temperature
h_bar = h/(2*pi)        # Reduced Planck's constant. h = Planck's constant
q                       # Elementary charge, roughly equal to 1.602e-19 C
V_T   = (kB*T)/q        # Thermal voltage
"""

import numpy as np
from scipy import integrate
import csv, time, random, sys
from pathlib import Path

# Define physical constants and parameters
Gamma = 0.2         # Rate constant of electron movement; normalised by 1/(beta*h_bar)
Cg = 200.0          # Gate capacitance of each NAND_Gate; normalised by q/V_T
alpha = 0.1         # Judgment threshold factor
V_D = 20.0           # Drain voltage, normalised by V_T
kT = 4.143e-21      # What we normalised energy with. Used to rescale energy dissipation to prevent overflow

# Define simulation parameters
tint = 100          # Time interval betweem data points; normalise by beta*h_bar
T = 2500000         # Total simulation time; normalised by beta*h_bar
Ntot = int(T/int)   # Number of timesteps

# Define number of chemical reactions, transition rate arrays, and cumulative sum transition rate arrays for NOT
Nreaction_NOT = 11
w_NOT = np.zeros(Nreaction_NOT)
wsum_NOT = np.zeros(Nreaction_NOT)

# Define current input sequence. Looping through all three arrays together will allow us to model for all possible current inputs
# E.g. for (Cin,Vin_A,Vin_B), index 0 is for (0,0,0), index 1 is for (0,0,1), index 2 is for (0,1,0), etc.
Vin_Cin_init = [0.0, 0.0, 0.0, 0.0, V_D, V_D, V_D, V_D]
Vin_A_init   = [0.0, 0.0, V_D, V_D, 0.0, 0.0, V_D, V_D]
Vin_B_init   = [0.0, V_D, 0.0, V_D, 0.0, V_D, 0.0, V_D]
N_INPUT = 8 # Total number of possible input combinations

# Define the initial states for each previous input. This is modelled using the theoretical ideal gate voltages for each gate
# Eg. for (Cin,Vin_A,Vin_B), index 0 shows theoretical ideal gate voltages at each logic gate for previous input (0,0,0), etc.
# INPUT = [000, 001, 010, 011, 100, 101, 110, 111]
Vg_XOR1_init = [0.0, V_D, V_D, 0.0, 0.0, V_D, V_D, 0.0]
Vg_XOR2_init = [0.0, V_D, V_D, 0.0, V_D, 0.0, 0.0, V_D]
Vg_AND1_init = [0.0, 0.0, 0.0, V_D, 0.0, 0.0, 0.0, V_D]
Vg_AND2_init = [0.0, 0.0, 0.0, 0.0, 0.0, V_D, V_D, 0.0]
Vg_OR_init   = [0.0, 0.0, 0.0, V_D, 0.0, V_D, V_D, V_D]

# Define the initial "microstates", i.e., the gate voltages of the intermediate NANDs and NOTs that compose each XOR, AND, and OR, for each previous input
# INPUT       = [000, 001, 010, 011, 100, 101, 110, 111]
Vg_XOR1_NAND1_init   = [V_D, V_D, V_D, 0.0, V_D, V_D, V_D, 0.0]
Vg_XOR1_NAND2_init   = [V_D, V_D, 0.0, V_D, V_D, V_D, 0.0, V_D]
Vg_XOR1_NAND3_init   = [V_D, 0.0, V_D, V_D, V_D, 0.0, V_D, V_D]
# Vg_XOR1_NAND4_init = [0.0, V_D, V_D, 0.0, 0.0, V_D, V_D, 0.0]

Vg_XOR2_NAND1_init   = [V_D, V_D, V_D, V_D, V_D, 0.0, 0.0, V_D]
Vg_XOR2_NAND2_init   = [V_D, 0.0, 0.0, V_D, V_D, V_D, V_D, V_D]
Vg_XOR2_NAND3_init   = [V_D, V_D, V_D, V_D, 0.0, V_D, V_D, V_D]
# Vg_XOR2_NAND4_init = [0.0, V_D, V_D, 0.0, V_D, 0.0, 0.0, V_D]

Vg_AND1_NAND_init    = [V_D, V_D, V_D, 0.0, V_D, V_D, V_D, 0.0]
# Vg_AND1_NOT_init   = [0.0, 0.0, 0.0, V_D, 0.0, 0.0, 0.0, V_D]
Vg_AND2_NAND_init    = [V_D, V_D, V_D, V_D, V_D, 0.0, 0.0, V_D]
# Vg_AND2_NOT_init   = [0.0, 0.0, 0.0, 0.0, 0.0, V_D, V_D, 0.0]

Vg_OR_NOT1_init      = [V_D, V_D, V_D, V_D, V_D, 0.0, 0.0, V_D]
Vg_OR_NOT2_init      = [V_D, V_D, V_D, 0.0, V_D, V_D, V_D, 0.0]
# Vg_OR_NAND_init    = [0.0, 0.0, 0.0, V_D, 0.0, V_D, V_D, V_D]
# Some of the voltage arrays are redundant, but useful for the user to visualise and double check

# Initialise variables to hold the actual gate voltages of each logic gate
Vg_XOR1 = np.zeros(4)
Vg_XOR2 = np.zeros(4)
Vg_AND1 = np.zeros(2)
Vg_AND2 = np.zeros(2)
Vg_OR   = np.zeros(3)

# Initialise output variables
# Dimension is as in (Prev_Input, Curr_Input, Timesteps)
Sum       = np.zeros((N_INPUT, N_INPUT, Ntot))         # Initialise sum output voltage vector
Cout      = np.zeros((N_INPUT, N_INPUT, Ntot))         # Initialise carry out output voltage vector
Qdiss     = np.zeros((N_INPUT, N_INPUT, Ntot))         # Initialise total energy dissipation vector
ErrorSum  = np.zeros((N_INPUT, N_INPUT, Ntot))         # Initialise sum error rate vector
ErrorCout = np.zeros((N_INPUT, N_INPUT, Ntot))         # Initialise carry out error rate vector

# Define Fermi, Bose, and Gauss distributions
def Fermi(e_diff):
    return 1.0/(np.exp(e_diff)+1)

def Bose(e_diff):
    # Bose tends to inf as its argument tends to 0, so we need to control it
    if e_diff > 1e-4:
        return 1.0/(np.exp(e_diff)-1)
    else:
        return 1e5
    
def Gauss(x, mu, var):
    return 1/np.sqrt(var*2*np.pi)*np.exp(-(x-mu)*(x-mu)/2/var)

# Define a function to convert non-negative integers less than 8 into a 3-bit binary representation
def convert_to_binary(x):
    binary = format(x, 'b')
    # Pad leftwards with 0s if needed:
    while len(binary) < 3:
        binary = "0" + binary
    return binary

# Model the time-evolution of voltages of a NAND gate
def NAND_propagation(Vin_X, Vin_Y, Vg):
    # State the total number of possible states the transistors in the NAND can have
    N_STATES = 16

    # Initialise variables to hold number of electrons in each transistor
    N_N1 = 0.0
    N_N2 = 0.0
    N_P1 = 0.0
    N_P2 = 0.0

    # Define number of chemical reactions, transition rate arrays, and cumulative sum transition rate arrays for NAND
    Nreaction_NAND = 21
    w_NAND = np.zeros(Nreaction_NAND)
    wsum_NAND = np.zeros(Nreaction_NAND)

    # Initialise electrode and transistor energy levels
    # These values have been normalised by q*V_T
    E_P1 = Vin_X
    E_P2 = Vin_Y
    E_N1 = 1.5*V_D - Vin_X
    E_N2 = 1.5*V_D - Vin_Y
    mu_s = 0.0
    mu_d = -V_D
    mu_g = -Vg

    # Calculate k values at each timestep
    # Those related to source and drain electrodes
    k_N2s=Gamma*Fermi(E_N2-mu_s)
    k_sN2=Gamma*(1.0-Fermi(E_N2-mu_s))
    k_P1d=Gamma*Fermi(E_P1-mu_d)
    k_dP1=Gamma*(1.0-Fermi(E_P1-mu_d))
    k_P2d=Gamma*Fermi(E_P2-mu_d)
    k_dP2=Gamma*(1.0-Fermi(E_P2-mu_d))

    # Those related to gate electrode
    # Because Vg and mu_g changes at each iteration, these k values do too
    k_P1g=Gamma*Fermi(E_P1-mu_g)
    k_gP1=Gamma*(1.0-Fermi(E_P1-mu_g))	
    k_P2g=Gamma*Fermi(E_P2-mu_g)
    k_gP2=Gamma*(1.0-Fermi(E_P2-mu_g))
    k_N1g=Gamma*Fermi(E_N1-mu_g)
    k_gN1=Gamma*(1.0-Fermi(E_N1-mu_g))

    # Those related to transfers between transistors
    if E_N1 > E_N2:
        k_N1N2=Gamma*Bose(E_N1-E_N2)
        k_N2N1=Gamma*(1+Bose(E_N1-E_N2))
    else:
        k_N2N1=Gamma*Bose(E_N2-E_N1)
        k_N1N2=Gamma*(1+Bose(E_N2-E_N1))
    if E_N1 > E_P1:
        k_N1P1=Gamma*Bose(E_N1-E_P1)
        k_P1N1=Gamma*(1+Bose(E_N1-E_P1))
    else:
        k_P1N1=Gamma*Bose(E_P1-E_N1)
        k_N1P1=Gamma*(1+Bose(E_P1-E_N1))
    if E_N1 > E_P2:
        k_N1P2=Gamma*Bose(E_N1-E_P2)
        k_P2N1=Gamma*(1+Bose(E_N1-E_P2))
    else:
        k_P2N1=Gamma*Bose(E_P2-E_N1)
        k_N1P2=Gamma*(1+Bose(E_P2-E_N1))
    if E_P1 > E_P2:
        k_P1P2=Gamma*Bose(E_P1-E_P2)
        k_P2P1=Gamma*(1+Bose(E_P1-E_P2))
    else:
        k_P2P1=Gamma*Bose(E_P2-E_P1)
        k_P1P2=Gamma*(1+Bose(E_P2-E_P1))

    # Initialise variables for current and time within the tint interval
    t = 0.0
    J1 = 0.0
    J2 = 0.0
    J3 = 0.0
    J4 = 0.0
    J5 = 0.0
    J6 = 0.0
    J7 = 0.0
    J8 = 0.0
    J9 = 0.0
    J10 = 0.0
    J11 = 0.0
    J12 = 0.0

    # While t is still within tint, find out when the next stochastic event will be
    while t < tint:
        # Source/Drain electrode transition rates
        w_NAND[1] = k_N2s*(1-N_N2)
        w_NAND[2] = k_sN2*N_N2
        w_NAND[3] = k_P1d*(1-N_P1)
        w_NAND[4] = k_dP1*N_P1
        w_NAND[5] = k_P2d*(1-N_P2)
        w_NAND[6] = k_dP2*N_P2
        # Gate electrode transition rates
        w_NAND[7] = k_P1g*(1-N_P1)
        w_NAND[8] = k_gP1*N_P1
        w_NAND[9] = k_P2g*(1-N_P2)
        w_NAND[10] = k_gP2*N_P2
        w_NAND[11] = k_N1g*(1-N_N1)
        w_NAND[12] = k_gN1*N_N1
        # Inter-transistor transition rates
        w_NAND[13]=k_N1N2*N_N2*(1-N_N1)
        w_NAND[14]=k_N2N1*N_N1*(1-N_N2)
        w_NAND[15]=k_N1P1*N_P1*(1-N_N1)
        w_NAND[16]=k_P1N1*N_N1*(1-N_P1)
        w_NAND[17]=k_N1P2*N_P2*(1-N_N1)
        w_NAND[18]=k_P2N1*N_N1*(1-N_P2)
        w_NAND[19]=k_P1P2*N_P2*(1-N_P1)
        w_NAND[20]=k_P2P1*N_P1*(1-N_P2)

        # Construct the cumulative sum transition rate array
        for j in range(1, Nreaction_NAND):
            wsum_NAND[j] = wsum_NAND[j-1] + w_NAND[j]
        wtot_NAND = wsum_NAND[Nreaction_NAND-1]     # Total transfer rate

        # Based on eqn A7, restructured in notes of week 2, pg 10
        # Monte Carlo Method
        dt = -np.log(random.uniform(0, 1))/wtot_NAND
        t += dt
        print("NAND", t)

        # Spin the wheel to see which electron gets to move
        p2 = random.uniform(0, 1)
        if p2 < wsum_NAND[1]/wtot_NAND:
            N_N2 = 1
            J1 += 1
        elif p2 < wsum_NAND[2]/wtot_NAND:
            N_N2 = 0
            J2 += 1
        elif p2 < wsum_NAND[3]/wtot_NAND:
            N_P1 = 1
            J3 += 1
        elif p2 < wsum_NAND[4]/wtot_NAND:
            N_P1 = 0
            J4 += 1
        elif p2 < wsum_NAND[5]/wtot_NAND:
            N_P2 = 1
            J5 += 1
        elif p2 < wsum_NAND[6]/wtot_NAND:
            N_P2 = 0
            J6 += 1
        elif p2 < wsum_NAND[7]/wtot_NAND:
            N_P1 = 1
            J7 += 1
        elif p2 < wsum_NAND[8]/wtot_NAND:
            N_P1 = 0
            J8 += 1
        elif p2 < wsum_NAND[9]/wtot_NAND:
            N_P2 = 1
            J9 += 1
        elif p2 < wsum_NAND[10]/wtot_NAND:
            N_P2 = 0
            J10 += 1
        elif p2 < wsum_NAND[11]/wtot_NAND:
            N_N1 = 1
            J11 += 1
        elif p2 < wsum_NAND[12]/wtot_NAND:
            N_N1 = 0
            J12 += 1
        elif p2 < wsum_NAND[13]/wtot_NAND:
            N_N1 = 1
            N_N2 = 0
        elif p2 < wsum_NAND[14]/wtot_NAND:
            N_N1 = 0
            N_N2 = 1
        elif p2 < wsum_NAND[15]/wtot_NAND:
            N_N1 = 1
            N_P1 = 0
        elif p2 < wsum_NAND[16]/wtot_NAND:
            N_N1 = 0
            N_P1 = 1
        elif p2 < wsum_NAND[17]/wtot_NAND:
            N_N1 = 1
            N_P2 = 0
        elif p2 < wsum_NAND[18]/wtot_NAND:
            N_N1 = 0
            N_P2 = 1
        elif p2 < wsum_NAND[19]/wtot_NAND:
            N_P1 = 1
            N_P2 = 0
        else:
            N_P1 = 0
            N_P2 = 1
        # End while loop

    # Net current to gate
    Jg=J8-J7+J12-J11+J10-J9

    # Calculate output voltage of NAND and additional energy dissipated by NAND within one timestep
    # Recall the definition of Jg used in the paper is the movement of negative charge
    Vg -= Jg*tint/Cg
    Qdiss_NAND = kT*(J1-J2)*tint*(mu_s-mu_g) + kT*(J3-J4+J5-J6)*tint*(mu_d-mu_g)
    return Vg, Qdiss_NAND

def NOT_propagation(Vin, Vg):
    # State the total number of possible states the transistors in the NOT can have
    N_STATES = 4

    # Initialise variables to hold number of electrons in each transistor
    N_N = 0.0
    N_P = 0.0

    # Define number of chemical reactions, transition rate arrays, and cumulative sum transistion arrays for NOT
    Nreaction_NOT = 11
    w_NOT = np.zeros(Nreaction_NOT)
    wsum_NOT = np.zeros(Nreaction_NOT)

    # Initialise the electrode and transistor energy levels
    E_P = Vin
    E_N = 1.5*V_D - Vin
    mu_s = 0.0
    mu_d = -V_D
    mu_g = -Vg

    # Calculate k values at each timestep
    # Those related to source and drain electrodes
    k_Ns=Gamma*Fermi(E_N-mu_s)
    k_sN=Gamma*(1.0-Fermi(E_N-mu_s))
    k_dP=Gamma*(1.0-Fermi(E_P-mu_d))
    k_Pd=Gamma*Fermi(E_P-mu_d)

    # Those related to gate electrode
    # Because Vg and mu_g changes at each iteration, these k values do too
    k_Ng=Gamma*Fermi(E_N-mu_g)
    k_gN=Gamma*(1.0-Fermi(E_N-mu_g))
    k_Pg=Gamma*Fermi(E_P-mu_g)
    k_gP=Gamma*(1.0-Fermi(E_P-mu_g))

    # Those related to transfers between transistors
    if E_N > E_P:
        k_NP=Gamma*Bose(E_N-E_P)
        k_PN=Gamma*(1+Bose(E_N-E_P))
    else:
        k_PN=Gamma*Bose(E_P-E_N)
        k_NP=Gamma*(1+Bose(E_P-E_N))

    # Initialise variables for current and time within the tint interval
    t = 0.0
    J1 = 0.0
    J2 = 0.0
    J3 = 0.0
    J4 = 0.0

    # While t is still within tint, find out when the next stochastic event will be
    while t < tint:
        # Source/Drain electrode transition rates
        w_NOT[1] = k_Ns*(1-N_N)
        w_NOT[2] = k_sN*N_N
        w_NOT[3] = k_Pd*(1-N_P)
        w_NOT[4] = k_dP*N_P
        # Gate electrode transition rates
        w_NOT[5] = k_Ng*(1-N_N)
        w_NOT[6] = k_gN*N_N
        w_NOT[7] = k_Pg*(1-N_P)
        w_NOT[8] = k_gP*N_P
        # Inter-transistor transition rates
        w_NOT[9] = k_PN*N_N*(1-N_P)
        w_NOT[10] = k_NP*N_P*(1-N_N)

        # Construct the cumulative sum transition rate array
        for j in range(1, Nreaction_NOT):
            wsum_NOT[j] = wsum_NOT[j-1] + w_NOT[j]
        wtot_NOT = wsum_NOT[Nreaction_NOT-1]        # Total transfer rate

        # Based on eqn A7, restructured in notes of week 2, pg 10
        # Monte Carlo Method
        dt = -np.log(random.uniform(0, 1))/wtot_NOT
        t += dt
        print("NOT", t)

        # Spin the wheel to see which electron gets to move
        p2 = random.uniform(0, 1)
        if p2 < wsum_NOT[1]/wtot_NOT:
            N_N = 1
            J1 += 1
        elif p2 < wsum_NOT[2]/wtot_NOT:
            N_N = 0
            J2 += 1
        elif p2 < wsum_NOT[3]/wtot_NOT:
            N_P = 1
            J3 += 1
        elif p2 < wsum_NOT[4]/wtot_NOT:
            N_P = 0
            J4 += 1
        elif p2 < wsum_NOT[5]/wtot_NOT:
            N_N = 1
            J5 += 1
        elif p2 < wsum_NOT[6]/wtot_NOT:
            N_N = 0
            J6 += 1
        elif p2 < wsum_NOT[7]/wtot_NOT:
            N_P = 1
            J7 += 1
        elif p2 < wsum_NOT[8]/wtot_NOT:
            N_P = 0
            J8 += 1
        elif p2 < wsum_NOT[9]/wtot_NOT:
            N_N = 0
            N_P = 1
        elif p2 < wsum_NOT[10]/wtot_NOT:
            N_N = 1
            N_P = 0
        # End while loop
        

    # Net current to gate
    Jg=J6-J5+J8-J7

    # Calculate output voltage of NOT and additional energy dissipated by NOT within one timestep
    # Recall the definition of Jg used in the paper is the movement of negative charge
    Vg -= Jg*tint/Cg
    Qdiss_NOT = kT*(J1-J2)*tint*(mu_s-mu_g) + kT*(J3-J4)*tint*(mu_d-mu_g)
    return Vg, Qdiss_NOT

def XOR_propagation(Vin_X, Vin_Y, Vg_NAND1, Vg_NAND2, Vg_NAND3, Vg_NAND4):
    # State number of NANDs in a XOR
    N_NANDS = 4

    # Create arrays to store input and output voltages of each NAND in the XOR
    VinX = np.zeros(N_NANDS)
    VinY = np.zeros(N_NANDS)
    Vg = np.array([Vg_NAND1, Vg_NAND2, Vg_NAND3, Vg_NAND4])

    VinX[0] = Vin_X
    VinY[0] = Vin_Y
    VinX[1] = Vin_X
    VinY[1] = Vg[0]
    VinX[2] = Vg[0]
    VinY[2] = Vin_Y
    VinX[3] = Vg[1]
    VinY[3] = Vg[2]

    # Initialise a variable to store the entire XOR (i.e., all 4 NANDS) energy dissipation within that timestep
    Qdiss_XOR = 0.0

    # Loop across all 4 NANDS within a XOR
    # Calculate the output gate voltage of each NAND and the additional energy dissipation of the whole XOR at each timestep
    for i in range(N_NANDS):
        Vg[i], Qdiss_NAND = NAND_propagation(VinX[i], VinY[i], Vg[i])
        Qdiss_XOR += Qdiss_NAND
    # XOR output is Vg[3], the rest are intermediate NAND gate voltages
    return Vg[0], Vg[1], Vg[2], Vg[3], Qdiss_XOR

def AND_propagation(Vin_X, Vin_Y, Vg_NAND, Vg_NOT):
    # An AND gate is created by having a NAND gate followed by a NOT gate
    
    # Initialise a variable to store AND energy dissipation within that timestep
    Qdiss_AND = 0.0

    # Propagate through the NAND
    Vg_NAND, Qdiss_NAND = NAND_propagation(Vin_X, Vin_Y, Vg_NAND)
    Qdiss_AND += Qdiss_NAND

    # Propagate through the NOT
    Vg_NOT, Qdiss_NOT = NOT_propagation(Vg_NAND, Vg_NOT)
    Qdiss_AND += Qdiss_NOT

    return Vg_NAND, Vg_NOT, Qdiss_AND


def OR_propagation(Vin_X, Vin_Y, Vg_NOT1, Vg_NOT2, Vg_NAND):
    # By De Morgan's Law, A+B = NOT(NOT(A)*NOT(B)) = NAND(NOT(A), NOT(B))
    # This means an OR gate is created by having two NOT gates, whose outputs
    # form the output of a subsequent NAND gate

    # Initialise a variable to store OR energy dissipation within that timestep
    Qdiss_OR = 0.0

    # Propagate through the two NOTs
    Vg_NOT1, Qdiss_NOT1 = NOT_propagation(Vin_X, Vg_NOT1)
    Qdiss_OR += Qdiss_NOT1
    Vg_NOT2, Qdiss_NOT2 = NOT_propagation(Vin_Y, Vg_NOT2)
    Qdiss_OR += Qdiss_NOT2

    # Propagate through the NAND
    Vg_NAND, Qdiss_NAND = NAND_propagation(Vg_NOT1, Vg_NOT2, Vg_NAND)
    Qdiss_OR += Qdiss_NAND

    return Vg_NOT1, Vg_NOT2, Vg_NAND, Qdiss_OR

def Full_Adder_propagation(Vin_Cin, Vin_A, Vin_B):
    # The full adder has two outputs: the sum S and the output carry Cout
    # S = A XOR B XOR Cin, Cout = AB + (A XOR B)Cin

    # Initialise a variable to store full adder energy dissipation within that timestep
    Qdiss_FA = 0.0

    # Propagate through XOR1
    Vg_XOR1[0], Vg_XOR1[1], Vg_XOR1[2], Vg_XOR1[3], Qdiss_XOR1 = XOR_propagation(Vin_A, Vin_B, Vg_XOR1[0], Vg_XOR1[1], Vg_XOR1[2], Vg_XOR1[3])
    Qdiss_FA += Qdiss_XOR1

    # Propagate through XOR2
    Vg_XOR2[0], Vg_XOR2[1], Vg_XOR2[2], Vg_XOR2[3], Qdiss_XOR2 = XOR_propagation(Vg_XOR1[3], Vin_Cin, Vg_XOR2[0], Vg_XOR2[1], Vg_XOR2[2], Vg_XOR2[3])
    Qdiss_FA += Qdiss_XOR2

    # Propagate through AND1
    Vg_AND1[0], Vg_AND1[1], Qdiss_AND1 = AND_propagation(Vin_A, Vin_B, Vg_AND1[0], Vg_AND1[1])
    Qdiss_FA += Qdiss_AND1

    # Propagate through AND2
    Vg_AND2[0], Vg_AND2[1], Qdiss_AND2 = AND_propagation(Vg_XOR1[3], Vin_Cin, Vg_AND2[0], Vg_AND2[1])
    Qdiss_FA += Qdiss_AND2

    # Propagate through OR
    Vg_OR[0], Vg_OR[1], Vg_OR[2], Qdiss_OR = OR_propagation(Vg_AND2[1], Vg_AND1[1], Vg_OR[0], Vg_OR[1], Vg_OR[2])
    Qdiss_FA += Qdiss_OR

    # Return Sum, Cout, and energy dissipated
    return Vg_XOR2[3], Vg_OR[2], Qdiss_FA

def main():
    start = time.time_ns()

    # Create folder to hold results and graphs
    output_folder = f"./V_D-{V_D}/Gillespie"
    try:
        Path(output_folder).mkdir()
    except FileExistsError:
        print("Directory already exists - no need to create again")

    # Create directories to hold results
    output_dir = f"{output_folder}/ResultsV_D-{V_D}"
    try:
        Path(output_dir).mkdir()
    except FileExistsError:
        print("Directory already exists - no need to create again")

    # Loop across all possible previous steady-state inputs
    for i in range(N_INPUT):
        # Initialise all gate voltages to the correct value for the given previous inputs
        Vg_XOR1[0], Vg_XOR1[1], Vg_XOR1[2], Vg_XOR1[3] = (Vg_XOR1_NAND1_init[i], Vg_XOR1_NAND2_init[i], Vg_XOR1_NAND3_init[i], Vg_XOR1_init[i])
        Vg_XOR2[0], Vg_XOR2[1], Vg_XOR2[2], Vg_XOR2[3] = (Vg_XOR2_NAND1_init[i], Vg_XOR2_NAND2_init[i], Vg_XOR2_NAND3_init[i], Vg_XOR2_init[i])
        Vg_AND1[0], Vg_AND1[1] = (Vg_AND1_NAND_init[i], Vg_AND1_init[i])
        Vg_AND2[0], Vg_AND2[1] = (Vg_AND2_NAND_init[i], Vg_AND2_init[i])
        Vg_OR[0], Vg_OR[1], Vg_OR[2] = (Vg_OR_NOT1_init[i], Vg_OR_NOT2_init[i], Vg_OR_init[i])

        # Loop across all possible current inputs
        for j in range(N_INPUT):
            print(f"Now at i = {i}, j = {j}")
            # Using Vin_Cin_init[j], Vin_A_init[j[]], Vin_B_init[j], we can loop through all the combination of current inputs easily
            # Loop across all timesteps:
            Qdiss_temp = 0.0
            for k in range(Ntot):
                Sum[i, j, k], Cout[i, j, k], Qdiss_additional = Full_Adder_propagation(Vin_Cin_init[j], Vin_A_init[j], Vin_B_init[j])
                Qdiss_temp += Qdiss_additional
                Qdiss[i, j, k] = Qdiss_temp
                # How we calculate the error depends on what the theoretical output of the combination should be
                var = 1/Cg      # Variance of the Gauss distribution to be used for integration
                if j in (1, 2, 4, 7):
                    # Sum is high
                    ErrorSum[i, j, k] = 1 - integrate.quad(Gauss, (1-alpha)*V_D, np.inf, args=(Sum[i, j, k], var))[0] # "The return value is a tuple, with the first element holding the estimated value of the integral and the second element holding an estimate of the absolute integration error"
                else:
                    # Sum is low
                    ErrorSum[i, j, k] = 1 - integrate.quad(Gauss, -np.inf, alpha*V_D, args=(Sum[i, j, k], var))[0]
                if j in (3, 5, 6, 7):
                    # Cout is high
                    ErrorCout[i, j, k] = 1 - integrate.quad(Gauss, (1-alpha)*V_D, np.inf, args=(Cout[i, j, k], var))[0]
                else:
                    # Cout is low
                    ErrorCout[i, j, k] = 1 - integrate.quad(Gauss, -np.inf, alpha*V_D, args=(Cout[i, j, k], var))[0]

            # Write results to a csv file
            with open(f"{output_dir}/Results-Prev{convert_to_binary(i)}-Curr{convert_to_binary(j)}.csv", "w") as file:
                writer = csv.DictWriter(file, fieldnames=["Timestep (s)", "Sum Voltage (V)",
                                                        "Carry Out Voltage (V)", "Sum Error Rate (Dimless)",
                                                        "Carry Out Error Rate (Dimless)", "Energy Dissipation (J)"],
                                                        lineterminator="\n")
                writer.writeheader()
                for k in range(Ntot):
                    writer.writerow({"Timestep (s)":k*tint, "Sum Voltage (V)":Sum[i, j, k],
                                    "Carry Out Voltage (V)":Cout[i, j, k], "Sum Error Rate (Dimless)":ErrorSum[i, j, k],
                                    "Carry Out Error Rate (Dimless)":ErrorCout[i, j, k], "Energy Dissipation (J)":Qdiss[i, j, k]})

    end = time.time_ns()

    elapsed = end - start

    print(f"Time elapsed is {elapsed/1e9:.2f} seconds")

if __name__ == "__main__":
    main()