"""
full_adder_deterministic.py

Usage: python full_adder_deterministic.py

This script models the time-evolution of the output voltage of a full-adder circuit.
The full-adder circuit is to be constructed using only NANDs and NOTs.
It uses a deterministic approach by applying the master equation dp/dt = Ap, whereby p is a vector holding probailities of each transistor holding an electron,
and A is the rate transfer matrix governing the movement of electrons in the system. We will solve for the steady-state response, i.e., dp/dt = 0 such that Ap = 0
The model takes into account the changes in behaviour if the previous input to the adder was different

ALL VARIABLES AND QUANTITIES USED ARE DIMENSIONLESS, UNLESS OTHERWISE STATED

The following quantities are not used in the code, but are required to understand the physics of the model:
beta  = 1/(kB*T)        # Coldness. kB = Boltzmann constant, T = absolute temperature
h_bar = h/(2*pi)        # Reduced Planck's constant. h = Planck's constant
q                       # Elementary charge, roughly equal to 1.602e-19 C
V_T   = (kB*T)/q        # Thermal voltage
"""

import numpy as np
from scipy.linalg import null_space
from scipy import integrate
import csv, time, sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=DeprecationWarning)

# Define physical constants and parameters
Gamma = 0.2         # Rate constant of electron movement; normalised by 1/(beta*h_bar)
Cg = 200.0          # Gate capacitance of each NAND_Gate; normalised by q/V_T
alpha = 0.1         # Judgment threshold factor
V_D = 2.5           # Drain voltage, normalised by V_T
kT = 4.143e-21      # What we normalised energy with. Used to rescale energy dissipation to prevent overflow

# Define simulation parameters
tint = 100          # Time interval between data points; normalised by beta*h_bar
T = 500000         # Total simulation time; normalised by beta*h_bar
Ntot = int(T/tint)   # Number of timesteps

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

    # Initialise the rate transfer matrix
    RNAND = np.zeros((N_STATES, N_STATES))

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

    # Populate the rate transfer matrix
    RNAND[1, 0]=k_P1d+k_P1g
    RNAND[2, 0]=k_P2d+k_P2g
    RNAND[3, 0]=k_N1g
    RNAND[4, 0]=k_N2s
    RNAND[0, 0]=-(RNAND[1, 0]+RNAND[2, 0]+RNAND[3, 0]+RNAND[4, 0])
    RNAND[0, 1]=k_dP1+k_gP1
    RNAND[2, 1]=k_P2P1
    RNAND[3, 1]=k_N1P1
    RNAND[5, 1]=k_P2d+k_P2g
    RNAND[6, 1]=k_N1g
    RNAND[7, 1]=k_N2s
    RNAND[1, 1]=-(RNAND[0, 1]+RNAND[2, 1]+RNAND[3, 1]+RNAND[5, 1]+RNAND[6, 1]+RNAND[7, 1])
    RNAND[0, 2]=k_dP2+k_gP2
    RNAND[1, 2]=k_P1P2
    RNAND[3, 2]=k_N1P2
    RNAND[5, 2]=k_P1d+k_P1g
    RNAND[8, 2]=k_N1g
    RNAND[9, 2]=k_N2s
    RNAND[2, 2]=-(RNAND[0, 2]+RNAND[1, 2]+RNAND[3, 2]+RNAND[5, 2]+RNAND[8, 2]+RNAND[9, 2])
    RNAND[0, 3]=k_gN1
    RNAND[1, 3]=k_P1N1
    RNAND[2, 3]=k_P2N1
    RNAND[4, 3]=k_N2N1
    RNAND[6, 3]=k_P1d+k_P1g
    RNAND[8, 3]=k_P2d+k_P2g
    RNAND[10, 3]=k_N2s
    RNAND[3, 3]=-(RNAND[0, 3]+RNAND[1, 3]+RNAND[2, 3]+RNAND[4, 3]+RNAND[6, 3]+RNAND[8, 3]+RNAND[10, 3])
    RNAND[0, 4]=k_sN2
    RNAND[3, 4]=k_N1N2
    RNAND[7, 4]=k_P1d+k_P1g
    RNAND[9, 4]=k_P2d+k_P2g
    RNAND[10, 4]=k_N1g
    RNAND[4, 4]=-(RNAND[0, 4]+RNAND[3, 4]+RNAND[7, 4]+RNAND[9, 4]+RNAND[10, 4])
    RNAND[1, 5]=k_dP2+k_gP2
    RNAND[2, 5]=k_dP1+k_gP1
    RNAND[6, 5]=k_N1P2
    RNAND[8, 5]=k_N1P1
    RNAND[13, 5]=k_N2s
    RNAND[14, 5]=k_N1g
    RNAND[5, 5]=-(RNAND[1, 5]+RNAND[2, 5]+RNAND[6, 5]+RNAND[8, 5]+RNAND[13, 5]+RNAND[14, 5])
    RNAND[1, 6]=k_gN1
    RNAND[3, 6]=k_dP1+k_gP1
    RNAND[5, 6]=k_P2N1
    RNAND[7, 6]=k_N2N1
    RNAND[8, 6]=k_P2P1
    RNAND[12, 6]=k_N2s
    RNAND[14, 6]=k_P2d+k_P2g
    RNAND[6, 6]=-(RNAND[1, 6]+RNAND[3, 6]+RNAND[5, 6]+RNAND[7, 6]+RNAND[8, 6]+RNAND[12, 6]+RNAND[14, 6])
    RNAND[1, 7]=k_sN2
    RNAND[4, 7]=k_dP1+k_gP1
    RNAND[6, 7]=k_N1N2
    RNAND[9, 7]=k_P2P1
    RNAND[10, 7]=k_N1P1
    RNAND[12, 7]=k_N1g
    RNAND[13, 7]=k_P2d+k_P2g
    RNAND[7, 7]=-(RNAND[1, 7]+RNAND[4, 7]+RNAND[6, 7]+RNAND[9, 7]+RNAND[10, 7]+RNAND[12, 7]+RNAND[13, 7])
    RNAND[2, 8]=k_gN1
    RNAND[3, 8]=k_gP2+k_dP2
    RNAND[5, 8]=k_P1N1
    RNAND[6, 8]=k_P1P2
    RNAND[9, 8]=k_N2N1
    RNAND[11, 8]=k_N2s
    RNAND[14, 8]=k_P1d+k_P1g
    RNAND[8, 8]=-(RNAND[2, 8]+RNAND[3, 8]+RNAND[5, 8]+RNAND[6, 8]+RNAND[9, 8]+RNAND[11, 8]+RNAND[14, 8])
    RNAND[2, 9]=k_sN2
    RNAND[4, 9]=k_dP2+k_gP2
    RNAND[7, 9]=k_P1P2
    RNAND[8, 9]=k_N1N2
    RNAND[10, 9]=k_N1P2
    RNAND[11, 9]=k_N1g
    RNAND[13, 9]=k_P1d+k_P1g
    RNAND[9, 9]=-(RNAND[2, 9]+RNAND[4, 9]+RNAND[7, 9]+RNAND[8, 9]+RNAND[10, 9]+RNAND[11, 9]+RNAND[13, 9])
    RNAND[3, 10]=k_sN2
    RNAND[4, 10]=k_gN1
    RNAND[7, 10]=k_P1N1
    RNAND[9, 10]=k_P2N1
    RNAND[11, 10]=k_P2d+k_P2g
    RNAND[12, 10]=k_P1d+k_P1g
    RNAND[10, 10]=-(RNAND[3, 10]+RNAND[4, 10]+RNAND[7, 10]+RNAND[9, 10]+RNAND[11, 10]+RNAND[12, 10])
    RNAND[8, 11]=k_sN2
    RNAND[9, 11]=k_gN1
    RNAND[10, 11]=k_dP2+k_gP2
    RNAND[12, 11]=k_P1P2
    RNAND[13, 11]=k_P1N1
    RNAND[15, 11]=k_P1d+k_P1g
    RNAND[11, 11]=-(RNAND[8, 11]+RNAND[9, 11]+RNAND[10, 11]+RNAND[12, 11]+RNAND[13, 11]+RNAND[15, 11])
    RNAND[6, 12]=k_sN2
    RNAND[7, 12]=k_gN1
    RNAND[10, 12]=k_dP1+k_gP1
    RNAND[11, 12]=k_P2P1
    RNAND[13, 12]=k_P2N1
    RNAND[15, 12]=k_P2d+k_P2g
    RNAND[12, 12]=-(RNAND[6, 12]+RNAND[7, 12]+RNAND[10, 12]+RNAND[11, 12]+RNAND[13, 12]+RNAND[15, 12])
    RNAND[5, 13]=k_sN2
    RNAND[7, 13]=k_dP2+k_gP2
    RNAND[9, 13]=k_dP1+k_gP1
    RNAND[11, 13]=k_N1P1
    RNAND[12, 13]=k_N1P2
    RNAND[14, 13]=k_N1N2
    RNAND[15, 13]=k_N1g
    RNAND[13, 13]=-(RNAND[5, 13]+RNAND[7, 13]+RNAND[9, 13]+RNAND[11, 13]+RNAND[12, 13]+RNAND[14, 13]+RNAND[15, 13])
    RNAND[5, 14]=k_gN1
    RNAND[6, 14]=k_dP2+k_gP2
    RNAND[8, 14]=k_dP1+k_gP1
    RNAND[13, 14]=k_N2N1
    RNAND[15, 14]=k_N2s
    RNAND[14, 14]=-(RNAND[5, 14]+RNAND[6, 14]+RNAND[8, 14]+RNAND[13, 14]+RNAND[15, 14])
    RNAND[11, 15]=k_dP1+k_gP1
    RNAND[12, 15]=k_dP2+k_gP2
    RNAND[13, 15]=k_gN1
    RNAND[14, 15]=k_sN2
    RNAND[15, 15]=-(RNAND[11, 15]+RNAND[12, 15]+RNAND[13, 15]+RNAND[14, 15])

    # Solve RNAND*p = 0 for p
    p = null_space(RNAND)

    # Normalise the p-vector using 1-norm
    psum = np.sum(p)
    for i in range(N_STATES):
        p[i] = p[i]/psum

    # Probability that a combination of transistors has electron(s)
    # The state vector p has the elements ordered in the unconventional way
    # 0000,
    # 0001, 0010, 0100, 1000,
    # 0011, 0101, 1001, 0110, 1010, 1100,
    # 1110, 1101, 1011, 0111,
    # 1111
    
    # LSB
    # |
    # |
    p_P1 = p[1] + p[5] + p[6] + p[7] + p[12] + p[13] + p[14] + p[15]		# 0001, 0011, 0101, 0111, 1001, 1011, 1101, 1111
    p_P2 = p[2] + p[5] + p[8] + p[9] + p[11] + p[13] + p[14] + p[15]		# 0010, 0011, 0110, 0111, 1010, 1011, 1110, 1111
    p_N1 = p[3] + p[6] + p[8] + p[10] + p[11] + p[12] + p[14] + p[15]	    # 0100, 0101, 0110, 0111, 1100, 1101, 1110, 1111
    p_N2 = p[4] + p[7] + p[9] + p[10] + p[11] + p[12] + p[13] + p[15]	    # 1000, 1001, 1010, 1011, 1100, 1101, 1110, 1111
    # |
    # v
    # MSB

    # Currents involving source and drain electrodes
    J1=k_N2s*(1-p_N2)
    J2=k_sN2*p_N2
    J3=k_P1d*(1-p_P1)
    J4=k_dP1*p_P1
    J5=k_P2d*(1-p_P2)
    J6=k_dP2*p_P2

    # Currents involving gate electrode
    J7=k_P1g*(1-p_P1)
    J8=k_gP1*p_P1
    J9=k_P2g*(1-p_P2)
    J10=k_gP2*p_P2
    J11=k_N1g*(1-p_N1)
    J12=k_gN1*p_N1

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

    # Initialise the rate transfer matrix
    RNOT = np.zeros((N_STATES, N_STATES))

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

    RNOT[1, 0]=k_Pd+k_Pg
    RNOT[2, 0]=k_Ns+k_Ng
    RNOT[0, 0]=-(RNOT[1, 0]+RNOT[2, 0])
    RNOT[0, 1]=k_dP+k_gP
    RNOT[2, 1]=k_NP
    RNOT[3, 1]=k_Ns+k_Ng
    RNOT[1, 1]=-(RNOT[0, 1]+RNOT[2, 1]+RNOT[3, 1])
    RNOT[0, 2]=k_sN+k_gN
    RNOT[1, 2]=k_PN
    RNOT[3, 2]=k_Pd+k_Pg
    RNOT[2, 2]=-(RNOT[0, 2]+RNOT[1, 2]+RNOT[3, 2])
    RNOT[1, 3]=k_sN+k_gN
    RNOT[2, 3]=k_dP+k_gP
    RNOT[3, 3]=-(RNOT[1, 3]+RNOT[2, 3])

    # Solve RNOT*p = 0 for p
    p = null_space(RNOT)

    # Normalise the p-vector using 1-norm
    psum = np.sum(p)
    for i in range(N_STATES):
        p[i] = p[i]/psum

    # Probability that a combination of transistors has electron(s)
    # The state vector p has elements ordered in this way
    # 00, 01, 10, 11

    # LSB
    # |
    # |
    p_P = p[1] + p[3]       # 01, 11
    p_N = p[2] + p[3]       # 10, 11
    # |
    # v
    # MSB

    # Currents involving source and drain
    J1=k_Ns*(1-p_N)
    J2=k_sN*p_N
    J3=k_Pd*(1-p_P)
    J4=k_dP*p_P

    # Currents involving gate electrode
    J5=k_Pg*(1-p_P)
    J6=k_gP*p_P
    J7=k_Ng*(1-p_N)
    J8=k_gN*p_N

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
    output_folder = f"./V_D-{V_D}"
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