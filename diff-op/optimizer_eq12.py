##########################################################################
# script to reproduce the results of J. Roeltgen et al 2025 NF 65 106020 #
##########################################################################

import sympy as sp
import numpy as np
from scipy.integrate import quad
from scipy.optimize import root_scalar
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True
import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.optimize import least_squares
from scipy.constants import e, m_e

##########################################################################
# it takes emissivity data of Hydrogen from openADAS, "plt96_h.dat", ##### 
##########################################################################

def load_adas_plt_h(filepath):
    """
    Parses a standard ADAS ADF11 (PLT) file for Hydrogen.
    Returns an interpolation function: f(log10_ne, log10_te)
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()

    # 1. Parse Grid Sizes (usually the first line)
    # Example: '  9  8' (9 Te points, 8 Ne points)
    header = lines[0].split()
    # print(header)
    n_ne = int(header[1])
    n_te = int(header[2])

    # Collect all numbers, skipping metadata
    raw_data = []
    for line in lines[1:]:
        if 'RADIATED POWERS' in line or 'OPEN-ADAS' in line:
            break
        # Clean comments of data and handle negative numbers properly
        clean_line = line.split('/')[0].replace('-', ' -')
        parts = clean_line.split()
        for p in parts:
            try:
                raw_data.append(float(p.replace('D', 'E')))
            except ValueError:
                continue

    # Slicing 
    log10_ne = np.array(raw_data[:n_ne])
    log10_te = np.array(raw_data[n_ne : n_ne + n_te])

    # Extract Matrix: shape (n_ne, n_te)
    data_start = n_ne + n_te
    matrix_vals = np.array(raw_data[data_start : data_start + (n_ne * n_te)])
    lz_matrix = matrix_vals.reshape((n_te, n_ne))
    
    # Create Interpolator: f(log10_ne, log10_te_ev)
    interp = RectBivariateSpline(log10_te, log10_ne, lz_matrix) 
    
    return interp

lz_interp = load_adas_plt_h("plt96_h.dat")

########################################################################
# function to get the emmisivity in Wm^3 given a electron ##############
# density ne in m^-3 and electron temperature te_ev in eV ##############
########################################################################

def get_lz_si(ne, te_ev, interp):
    """
    ne: density in m^-3
    te_ev: temperature in eV
    returns: Lz in W * m^3
    """
    # 1. Evaluate log10(W*cm^3)
    # ADAS ne is usually in cm^-3 (10^13 to 10^20 is typical)
    # If your input ne is m^-3, convert to cm^-3 for the lookup
    ne_cm3 = ne * 1e-6
    log_lz = interp(np.log10(te_ev), np.log10(ne_cm3))[0,0]
    
    # 2. Convert result to W*m^3
    # 10**log_lz gives W*cm^3. Multiply by 1e-6 to get W*m^3
    return (10**log_lz) * 1e-6


#########################################################
##### set up of equation 12 of the paper 
#########################################################

# 1. Define the integrand 

def integrand(v_bar, Te, A_bar, alpha, beta, gamma, V0):
    # In section 2.2 variable transformations, eq. 5 is scaled, first step is to define 
    # v_bar = v*sqrt(m_e/2e)
    # and therefore, x = v/V0 --> x = v_bar/(V0*((m_e/(2*e))**(1/2)))
    # it return the expression to be integrated: (v_bar^4 / Te^(3/2)) * nu(v_bar) * exp(-v_bar^2 / Te)
    # and it caculates nu(v_bar) = A_bar*((alpha + beta)/(beta*(x**(-alpha)) + alpha*(x**(beta))))*(v_bar**gamma)
    x = v_bar/V0
    return ((v_bar**4)/(Te**(3/2)))*A_bar*((alpha + beta)/(beta*(x**(-alpha)) + alpha*(x**(beta))))*(v_bar**gamma)*np.exp(-(v_bar**2)/Te)

# the following function, is the same as the one above, but incorporates 
# some specific return values based on the asymptotic form of nu(v)

def safe_integrand(v_bar, Te, A_bar, alpha, beta, gamma, V0):

    # same definition as in integrand
    x = v_bar / V0
    
    # Handles absolute zero to prevent 0.0 ** negative_power
    if x == 0.0:
        return 0.0
        
    # Handles asymptotic behaviour of nu (eq. 7)
    if x < 1.0:
        # Multiply top and bottom by x^alpha (v -> 0)
        num = (alpha + beta) * (x ** alpha)
        den = beta + alpha * (x ** (alpha + beta))
        fraction = num / den
    else:
        # Multiply top and bottom by x^-beta (v -> inf)
        num = (alpha + beta) * (x ** -beta)
        den = beta * (x ** -(alpha + beta)) + alpha
        fraction = num / den

        
    # Construct the full integrand from eq. 12 or 13
    term1 = (v_bar ** 4) / (Te ** 1.5)
    term2 = A_bar * fraction * (v_bar ** gamma)
    term3 = np.exp(-(v_bar ** 2) / Te)
    
    return term1 * term2 * term3


# this function is the function to be minimized, as in eq. 12 however
# a couple of modifications were done, in concrete the factor (1/C)
# had to be taken out to obtain the same orders of magnitude as emissivity data
# the constrains are also apply in this function by means of the function returning
# a large penalty, see below
def objective_function(params, Te_data, Li_data, weight_w, C_constant):
    # this function takes arrays of size j Te_data and the corresponding Li_data 
    # computes rhs of eq. 13, and return array with differences in eq. 12
    # not yet squared

    # Unpack the parameters the optimizer is currently guessing (nu parameters)
    A_bar, alpha, beta, gamma, V0 = params

    # --- ENFORCE CONSTRAINTS HERE ---
    # If the optimizer guesses an invalid combination, return a massive error
    # in this case the constrains are: alpha + gamma > 0, gamma - beta < 2
    if (alpha + gamma <= 0) or (gamma - beta >= 2):
        # Return an array of huge numbers to tell the optimizer "Bad guess!"
        return np.ones_like(Te_data) * 1e10
    
    # in here the "correct" sign for alpha, gamma and beta is impose
    # take into account that this is known from already optimized values
    # for Hydrogen in the paper
    if alpha < 0:
        # Return an array of huge numbers to tell the optimizer "Bad guess!"
        return np.ones_like(Te_data) * 1e10
    if gamma > 0:
        # Return an array of huge numbers to tell the optimizer "Bad guess!"
        return np.ones_like(Te_data) * 1e10
    if beta < 0:
        # Return an array of huge numbers to tell the optimizer "Bad guess!"
        return np.ones_like(Te_data) * 1e10
    
    residuals = []

    # according to eq. 12 the integrand should be multiply by 1/C, with C being
    # C_constant = ((8*e)/(np.sqrt(np.pi)*n_i))*(((2*e)/m_e)**(gamma/2))
    # from eq. 11
    # however to get the same values obtained by J. Roeltgen for A_bar, 1/C factor must not be included
    # meaning that A_bar is optimized to absorves all the constants
    # i am not sure of this, but this is what heuristically work for me

    # 1. The prefactor: 8e / (sqrt(pi) * n_i)
    prefactor = (8 * e) / (np.sqrt(np.pi) * n_i)
    
    # 2. The velocity/mass factor: (2e / m_e)^(gamma / 2)
    velocity_factor = ((2 * e) / m_e) ** (gamma / 2)

    # C_constant = prefactor * velocity_factor
    
    # computing integral 
    for j in range(len(Te_data)):
        Te_j = Te_data[j]
        Li_j = Li_data[j]
        
        # Calculate the integral for this specific temperature using scipy's quad
        # args=(...) passes the extra variables to the integrand function
        integral_val, error = quad(safe_integrand, 0, np.inf, 
                                   args=(Te_j, A_bar, alpha, beta, gamma, V0))

        # Calculate the modeled emissivity
        # in this case C_cosntat is 1
        C_constant = 1
        # and it is multiply by Bs to scaled the values
        model_emissivity = (1/C_constant) * integral_val  * Bs
        
        # Calculate the weighted residual for this temperature point
        # here Li_j is multiply also by Bs, and the residual is multiply by (Bs*Li_j)**weight_w - 1
        # to weight for Te Li values with significant magnitude
        residual = (model_emissivity - Li_j*Bs) * ((Bs*Li_j) ** (weight_w - 1))
        residuals.append(residual)

    # the function below least_squares will take care of the optimisation  
    # least_squares expects an array of residuals (it squares and sums them internally)
    return np.array(residuals)

# main code

Bs = 1e30
n_i = 5e19
weight_w = 2
C_constant = 1 # it needs gamma so it is evaluated inside objective loop 

# generating Li(Te) array
Te_data = np.geomspace(0.4, 1e3, 30)
Li_data = np.array([get_lz_si(1e19, te, lz_interp) for te in Te_data])

# initial guess of the parameters A_bar, alpha, beta, gamma, V0
# take into account that a good guess was done given that values were already known
initial_guess = [1e-32, 8e3, 1.0, -1.0, 2.0] 
# estimate of the order of magnitude of the parameters 
parameter_scales = [1e-32, 1e3, 1.0, 1.0, 1.0]

# function that does the optimisation, tolerances xtol, ftol, gtol were decrease
result = least_squares(objective_function, initial_guess, args=(Te_data, Li_data, weight_w, C_constant),
                        verbose = 2, xtol = 1e-15, x_scale=parameter_scales, ftol = 1e-16, gtol = 1e-16) # , diff_step = 1e-4
print("A_bar, alpha, beta, gamma, V0")
print("Obtained Parameters:", result.x)

# computing Li according to equation 13 with the found parameters:

A_bar, alpha, beta, gamma, V0 = result.x

Operator_arr = np.zeros_like(Te_data)

for i in range(len(Te_data)):
    Te_j = Te_data[i] 
    integral_val, error = quad(safe_integrand, 0, np.inf, 
                                   args=(Te_j, A_bar, alpha, beta, gamma, V0))

    Operator_arr[i] = integral_val

# the following plot reproduces figure 1 of the paper for H+0
plt.plot(np.log10(Te_data), np.log10(Li_data), '-b', label = 'OpenADAS')
plt.plot(np.log10(Te_data), np.log10(Operator_arr), '*-g', label = 'My fit')

# parameters obtain by author from the first row of table 2.
A_bar = 5.5949e-32
alpha = 8e3
beta = 7.9587517e-1
gamma = -1.391
V0 = 3.5201735
print("Parameters from paper:", [A_bar, alpha, beta, gamma, V0])

Operator_arr = np.zeros_like(Te_data)
for i in range(len(Te_data)):
    Te_j = Te_data[i] 
    integral_val, error = quad(safe_integrand, 0, np.inf, 
                                   args=(Te_j, A_bar, alpha, beta, gamma, V0))

    Operator_arr[i] = integral_val

plt.plot(np.log10(Te_data), np.log10(Operator_arr), '*r', label = 'Roeltgen fit')
# plt.xlabel(r'log($T_e$ [eV])')
plt.xlabel('log10[ $T_{e} ($eV$)$ ]', size = 14)
plt.ylabel('log10[ Emissivity ($Wm^3$) ]', size = 14)
plt.legend()
plt.savefig('fits.png')
plt.show()

