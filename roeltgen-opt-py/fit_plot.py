# script to fit and showing plots of parameters

import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt
import argparse

# Import from your other modules
from data_parser import load_adas_plt_h, get_lz_si
from data_parser import load_roeltgen_formatted
from optimizer_core import run_single_optimization, safe_integrand
from error_analysis import error_analysis

def get_model_emissivity(params, Te_data):
    """Evaluates the final optimized integral over all temperatures."""
    A_scaled, alpha, beta, V0, gamma = params
    #alpha = alpha * 1e3 # Remember to scale alpha back up for the integral
    calcY = np.zeros_like(Te_data)
    
    for j, Te_j in enumerate(Te_data):
        # val, _ = quad(safe_integrand, 0, np.inf, 
        #               args=(Te_j, A_scaled, alpha, beta, V0, gamma),
        #               points=[V0], epsabs=1e-8, epsrel=1e-8)
        # calcY[j] = val
        # FIX: Match the split integral from the optimizer core

        v_max = np.sqrt(30.0 * Te_j)

        if V0 < v_max:

            val_1, _ = quad(safe_integrand, 0, V0, 
                            args=(Te_j, A_scaled, alpha, beta, V0, gamma),
                            epsabs=1e-8, epsrel=1e-8)
            val_2, _ = quad(safe_integrand, V0, v_max, 
                            args=(Te_j, A_scaled, alpha, beta, V0, gamma),
                            epsabs=1e-8, epsrel=1e-8)
            calcY[j] = val_1 + val_2

        else:
            
            val, _ = quad(safe_integrand, 0, v_max, 
                            args=(Te_j, A_scaled, alpha, beta, V0, gamma),
                            epsabs=1e-8, epsrel=1e-8)
            calcY[j] = val
    return calcY

def plot_fit(params_fit, Te_data, target_data_scaled, species, charge_state, w):

    calcY_scaled = get_model_emissivity(params_fit, Te_data)

    # the following plot reproduces figure 1 of the paper for H+0
    plt.plot(np.log10(Te_data), target_data_scaled/Bs, '-b', label = 'OpenADAS')
    plt.plot(np.log10(Te_data), calcY_scaled/Bs, '*-g', label = 'My fit')

    A_phys = params_fit[0] / Bs
    alpha = params_fit[1] # Remember to scale alpha back up for the integral
    beta = params_fit[2]
    V0 = params_fit[3]
    gamma = params_fit[4] 

    # 2. Create the parameter string
    param_text = (
        fr"Fitted Parameters:" "\n"
        fr"$A = {A_phys:.4e}$" "\n"
        fr"$\alpha = {alpha:.4f}$" "\n"
        fr"$\beta = {beta:.4f}$" "\n"
        fr"$V_0 = {V0:.4f}$" "\n"
        fr"$\gamma = {gamma:.4f}$"
    )

    plt.text(0.65, 0.55, param_text, 
        transform=plt.gca().transAxes, 
        fontsize=11, 
        verticalalignment='top', 
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))
    
    if species == 'H' and charge_state == '0':
        roeltgen_params_model = [5.5949e-32, 8.0000102e3, 7.9587517e-1, 3.5201735, -1.3919964]
    elif species == 'Li' and charge_state == '0':
        roeltgen_params_model = [3.2276775e-31, 8e3, 1.6314718, 1.8314244, -7.2283424] # Placeholder values for Li+0
    elif species == 'Li' and charge_state == '1':
        roeltgen_params_model = [1.8391597e-32, 1.5550315e4, 4.8963899e-1, 7.7295814, -1.4801717] # Placeholder values for Li+1
    elif species == 'Li' and charge_state == '2':
        roeltgen_params_model = [1.1269619e-32, 6.0237296e3, 8.8021499e-1, 9.9228372, -1.2597055] # Placeholder values for Li+2
    elif species == 'He' and charge_state == '0':
        roeltgen_params_model = [8.0128134e-33, 8e3, 6.3932130e-1, 4.8535296, -1.0357297] # Placeholder values for He+0
    elif species == 'He' and charge_state == '1':
        roeltgen_params_model = [4.0872258e-32, 8e3, 5.3427114e-1, 6.6810820, -1.6390255] # Placeholder values for He+1
    
    roeltgen_model = get_model_emissivity(roeltgen_params_model, Te_data)

    param_text = (
        fr"Roeltgen Parameters:" "\n"
        fr"$A = {roeltgen_params_model[0]:.4e}$" "\n"
        fr"$\alpha = {roeltgen_params_model[1]:.4f}$" "\n"
        fr"$\beta = {roeltgen_params_model[2]:.4f}$" "\n"
        fr"$V_0 = {roeltgen_params_model[3]:.4f}$" "\n"
        fr"$\gamma = {roeltgen_params_model[4]:.4f}$"
    )

    plt.text(0.2, 0.55, param_text, 
        transform=plt.gca().transAxes, 
        fontsize=11, 
        verticalalignment='top', 
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))

    plt.plot(np.log10(Te_data), roeltgen_model, '*r', label = 'Roeltgen fit')
    plt.xlabel('log10[ $T_{e} ($eV$)$ ]', size = 14)
    plt.ylabel('log10[ Emissivity ($Wm^3$) ]', size = 14)
    plt.legend(loc='lower right')
    plt.title(f'{species} + {charge_state} Fit with Weight Power = {w:.2f}', size = 16)
    #plt.savefig('fits.png')
    plt.show()

    return 0

if __name__ == "__main__":

# --- ADD ARGPARSE ---
    parser = argparse.ArgumentParser(description="Run the Radiation Fitter")
    parser.add_argument('--optimizer', type=str, default='ipopt', 
                        choices=['slsqp', 'ipopt', 'fmincon'],
                        help="Choose the optimization engine: slsqp, ipopt, or fmincon")
    
    # --- ADD SPECIES AND CHARGE ARGUMENTS ---
    parser.add_argument('--species', type=str, default='H',
                        help="Element symbol (e.g., H, He, Li)")
    parser.add_argument('--charge', type=int, default=0,
                        help="Charge state integer (e.g., 1, 2, 3)")
    
    # add argument to wether plot the fit or not
    parser.add_argument('--plot', type=bool, default=True)

    args = parser.parse_args()

    print("=========================================")
    print(f"--- FITTING SPECIES: {args.species.capitalize()}^{args.charge}+ ---")
    print(f"--- OPTIMIZER: {args.optimizer.upper()} ---")
    print("=========================================\n")

    # --- START MATLAB ENGINE ONCE ---
    eng = None
    if args.optimizer == 'fmincon':
        print("Booting MATLAB Engine (This takes ~15 seconds)...")
        import matlab.engine
        eng = matlab.engine.start_matlab()
        print("MATLAB Engine connected!")
    
    # 1. Load Data using the new parser
    try:
        # Note: make sure your files are saved in a folder named 'data'
        # in the same directory as the script.
        Te_data, target_data_unscaled = load_roeltgen_formatted(
            species=args.species, 
            charge_state=args.charge, 
            log10_ne_cm3=13.0
        )
    except Exception as e:
        print(f"Data Loading Error: {e}")
        exit(1)

    try:

        print("--- Starting Radiation Fitter ---")
        
        # 1. Load Data
        # For Hydrogen 
        species = args.species
        #charge_state = str(int(args.charge)-1)
        charge_state = str(args.charge)

        print(charge_state)
        # interp = load_adas_plt_h("plt_data/plt96_h.dat")

        # Te_data = 10 ** interp.get_knots()[0][3:-3]
        
        Bs = 1e30
        # target_data_unscaled = np.array([get_lz_si(1e19, te, interp) for te in Te_data])
        target_data_scaled = target_data_unscaled * Bs
        
        # 2. Setup Loop Parameters
        # Order: [A_scaled, alpha, beta, V0, gamma]
        initial_guess = [0.02, 8e3, 0.8, 4.0, -4.0] 
        #initial_guess = [5.5949e-32*Bs, 8, 7.9587517e-1, 3.52, -1.391] # Initial guess for the 5 parameters
        weight_powers = np.arange(0.1, 0.31, 0.01) 
        print("Weight Powers to Test:", weight_powers)
        best_fit_params = None
        best_weight = None
        global_min_error = np.inf
        
        # 3. OUTER LOOP: Iterate through the weight powers
        for w in weight_powers:
            print(f"\nTesting Weight Power: {w:.2f}")
            
            # Create a working copy of the guess so we can modify it in the inner loop
            current_guess = list(initial_guess)
            # print(current_guess)
            # exit()
            passed_all_tests = False
            
            # 4. INNER LOOP: The V0 Kick (Equivalent to MATLAB's while loop)
            while current_guess[3] < 20 and not passed_all_tests:
                print(f"  -> Running optimizer with V0 guess = {current_guess[3]:.2f}")
                
                result = run_single_optimization(current_guess, Te_data, target_data_scaled, w, optimizer_choice=args.optimizer, eng=eng)

                #plot_fit(result.x, Te_data, target_data_scaled, species, w)
                
                if result.success:
                    print("Optimizer result plotting result:")

                    if args.plot:
                        plot_fit(result.x, Te_data, target_data_scaled, species, charge_state, w)

                    calcY_scaled = get_model_emissivity(result.x, Te_data)
                    #calcY_scaled = get_model_emissivity(initial_guess, Te_data)
                    ratio = np.maximum(calcY_scaled / target_data_scaled, target_data_scaled / calcY_scaled)
                    
                    # Send to Error Analysis Judge
                    successes, max_error = error_analysis(ratio, Te_data, target_data_scaled)

                    print("------------Fit performance-----------------")
                    print("--------------------------------------------")
                    print("------------ ratio of model to target: -------------")
                    print(ratio)
                    print("------------ successes: -------------")
                    print(successes)
                    print("------------ max error in each zone: -------------")
                    print(max_error)
                    print("--------------------------------------------")
                    
                    if all(successes):
                        print("    [PASS] Fit meets all physical criteria!")
                        passed_all_tests = True
                        
                        # Check if this is the best fit overall
                        if max_error[0] < global_min_error:
                            global_min_error = max_error[0]
                            best_fit_params = result.x
                            best_weight = w
                            print("    --> New Global Best Fit Found!")
                            
                        # Chain this successful fit as the starting point for the next weight
                        initial_guess = result.x
                    else:
                        print("    [FAIL] Fit criteria not met. Kicking V0 by 2.00x and retrying...")
                        current_guess[3] *= 2.0 # Increase V0 just like MATLAB
                else:
                    print("    [FAIL] Optimizer did not converge. Kicking V0 by 2.00x and retrying...")
                    current_guess[3] *= 2.0
                    
            if not passed_all_tests:
                print(f"  Giving up on weight {w:.2f}. V0 exceeded 200.")

        # 5. Final Output
        print("\n=========================================")
        if best_fit_params is not None:
            print(f"FINAL BEST FIT (Weight = {best_weight:.2f}):")
            A_phys = best_fit_params[0] / Bs
            print(f"A_phys = {A_phys:.4e}")
            print(f"alpha  = {best_fit_params[1]:.4f}")
            print(f"beta   = {best_fit_params[2]:.4f}")
            print(f"V0     = {best_fit_params[3]:.4f}")
            print(f"gamma  = {best_fit_params[4]:.4f}")
        else:
            print("No fit passed all error criteria across all weights.")


    finally:
        # This block executes unconditionally, cleanly returning the license
        if eng is not None:
            print("Shutting down MATLAB engine and releasing license...")
            eng.quit()