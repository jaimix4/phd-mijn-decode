import numpy as np 

e = 1.6e-19 # elementary charge in Coulombs
m_e = 9.11e-31 # electron mass in kg
ni = 1e19 # ion density in m^-3

gamma = -1.5

C = (8*e/(np.sqrt(np.pi)*ni))*(2*e/(m_e))**(gamma/2)

print(f"C = {C:.3e} m^3/s^(gamma/2)")
