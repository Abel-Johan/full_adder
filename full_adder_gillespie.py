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