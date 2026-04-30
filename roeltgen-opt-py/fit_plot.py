# script to fit and showing plots of parameters
import os
import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt
import argparse
from datetime import datetime # Added for unique run ID

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

        # instead of integrating to infinity, I integrate to a large velocity. 
        v_max = np.sqrt(40.0 * Te_j)

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
# He+1    
# FINAL BEST FIT (Weight = 0.14):
# A_phys = 6.3245e-32
# alpha  = 7919.1261
# beta   = 0.2845
# V0     = 6.6232
# gamma  = -1.8788

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

def plot_and_save_fit(params_fit, Te_data, target_data_scaled, species, charge_state, density_log10, w, run_id, optimizer, Bs, show_plot=False, save_plot=True):
    """Generates a 1x2 subplot, showing it interactively and/or saving it."""
    calcY_scaled = get_model_emissivity(params_fit, Te_data)
    
    os.makedirs("results", exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    x_log = np.log10(Te_data)
    y_target = target_data_scaled / Bs
    y_model = calcY_scaled / Bs

    # --- 1. Base Plots (Target and My Fit) ---
    for ax in (ax1, ax2):
        ax.plot(x_log, y_target, '-b', label='OpenADAS')
        ax.plot(x_log, y_model, '*-g', label='My fit')
        ax.set_xlabel('log10[ T_e (eV) ]', fontsize=12)
        ax.grid(True, alpha=0.3)

    ax1.set_ylabel('log10[ Emissivity (Wm^3) ]', fontsize=12)
    ax2.set_ylabel('Emissivity (Wm^3)', fontsize=12)
    ax1.set_title('Log-Log Scale', fontsize=14)
    ax2.set_title('Linear-Log Scale', fontsize=14)

    # --- 2. Roeltgen Parameters & Plotting ---
    roeltgen_params_model = None
    if species == 'H' and charge_state == '0':
        roeltgen_params_model = [5.5949e-32, 8.0000102e3, 7.9587517e-1, 3.5201735, -1.3919964]
    elif species == 'Li' and charge_state == '0':
        roeltgen_params_model = [3.2276775e-31, 8e3, 1.6314718, 1.8314244, -7.2283424] 
    elif species == 'Li' and charge_state == '1':
        roeltgen_params_model = [1.8391597e-32, 1.5550315e4, 4.8963899e-1, 7.7295814, -1.4801717] 
    elif species == 'Li' and charge_state == '2':
        roeltgen_params_model = [1.1269619e-32, 6.0237296e3, 8.8021499e-1, 9.9228372, -1.2597055] 
    elif species == 'He' and charge_state == '0':
        roeltgen_params_model = [8.0128134e-33, 8e3, 6.3932130e-1, 4.8535296, -1.0357297] 
    elif species == 'He' and charge_state == '1':
        roeltgen_params_model = [4.0872258e-32, 8e3, 5.3427114e-1, 6.6810820, -1.6390255] 

    if roeltgen_params_model is not None:
        # Scale A up so get_model_emissivity behaves consistently, then scale the result back down
        r_params_scaled = list(roeltgen_params_model)
        r_params_scaled[0] *= Bs 
        roeltgen_model = get_model_emissivity(r_params_scaled, Te_data) / Bs

        ax1.plot(x_log, np.log10(roeltgen_model), '*r', label='Roeltgen fit')
        ax2.plot(x_log, roeltgen_model, '*r', label='Roeltgen fit')

        r_text = (
            f"Roeltgen Parameters:\n"
            f"A = {roeltgen_params_model[0]:.4e}\n"
            f"alpha = {roeltgen_params_model[1]:.4f}\n"
            f"beta = {roeltgen_params_model[2]:.4f}\n"
            f"V0 = {roeltgen_params_model[3]:.4f}\n"
            f"gamma = {roeltgen_params_model[4]:.4f}"
        )
        ax1.text(0.05, 0.95, r_text, transform=ax1.transAxes, fontsize=11, 
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))
    else:
        # If we plot a log scale, we must log the data arrays
        ax1.plot(x_log, np.log10(y_target), '-b', label='OpenADAS')
        ax1.plot(x_log, np.log10(y_model), '*-g', label='My fit')

    # Re-apply log-log data to ax1 (to overwrite the linear data drawn in the base loop)
    ax1.lines[0].set_ydata(np.log10(y_target))
    ax1.lines[1].set_ydata(np.log10(y_model))

    # --- 3. My Fit Parameters & Legends ---
    A_phys = params_fit[0] / Bs
    param_text = (
        f"Fitted Parameters:\n"
        f"A = {A_phys:.4e}\n"
        f"alpha = {params_fit[1]:.4f}\n"
        f"beta = {params_fit[2]:.4f}\n"
        f"V0 = {params_fit[3]:.4f}\n"
        f"gamma = {params_fit[4]:.4f}"
    )
    ax2.text(0.05, 0.95, param_text, transform=ax2.transAxes, fontsize=11, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))

    ax1.legend(loc='lower right')
    ax2.legend(loc='lower right')
    plt.suptitle(f"Run ID: {run_id} | {species}^{charge_state}+ | n_e = 10^{density_log10} | w = {w:.2f} | Opt: {optimizer.upper()}", fontsize=16)

    # --- 4. Show and Save Actions ---
    if show_plot:
        plt.show() # Note: This will pause the terminal until the plot window is closed
        
    if save_plot:
        filename = f"results/{species}_{charge_state}_{run_id}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"    -> Plot saved to {filename}")
        
    if not show_plot:
        plt.close() # Clean up memory if we only saved it

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
    #parser.add_argument('--plot', type=bool, default=True)

    parser.add_argument('--plot', action='store_true', 
                        help="Show interactive plots whenever a better fit is found")

    args = parser.parse_args()

    # Generate unique run ID
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    density_log10 = 13.0 # Hardcoded for now, can be an argument later
    Bs = 1e30

    print("=========================================")
    print(f"--- FITTING SPECIES: {args.species.capitalize()}^{args.charge}+ ---")
    print(f"--- OPTIMIZER: {args.optimizer.upper()} ---")
    print(f"--- RUN ID: {run_id} ---")
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
            log10_ne_cm3=density_log10
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

        # target_data_unscaled = np.array([get_lz_si(1e19, te, interp) for te in Te_data])
        target_data_scaled = target_data_unscaled * Bs
        
        # 2. Setup Loop Parameters
        # Order: [A_scaled, alpha, beta, V0, gamma]
        initial_guess = [0.02, 8e3, 0.8, 1.5, -4.0] 
        #initial_guess = [5.5949e-32*Bs, 8, 7.9587517e-1, 3.52, -1.391] # Initial guess for the 5 parameters
        weight_powers = np.arange(0.1, 0.20, 0.01) 
        print("Weight Powers to Test:", weight_powers)

        # best_fit_params = None
        # best_weight = None
        # global_min_error = np.inf

        global_min_error = np.inf
        best_fit_params = None
        best_weight = None
        best_successes = None
        best_max_error = None
        
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
                    calcY_scaled = get_model_emissivity(result.x, Te_data)
                    ratio = np.maximum(calcY_scaled / target_data_scaled, target_data_scaled / calcY_scaled)
                    successes, max_error = error_analysis(ratio, Te_data, target_data_scaled)

                    print(f"  V0={current_guess[3]:.2f} | Z1 Err: {max_error[0]:.2f} | Z2 Err: {max_error[1]:.2f} | Z3 Err: {max_error[2]:.2f} | Passed: {sum(successes)}/6")
                    
                    if max_error[0] < global_min_error:
                        global_min_error = max_error[0]
                        best_fit_params = result.x
                        best_weight = w
                        best_successes = successes
                        best_max_error = max_error
                        print(f"    --> (New Global Best Fit Logged)")
                        
                        # --- INTERACTIVE DISPLAY TRIGGER ---
                        if args.plot:
                            print("    --> Showing intermediate best plot (Close window to continue)...")
                            plot_and_save_fit(best_fit_params, Te_data, target_data_scaled, species, charge_state, density_log10, best_weight, run_id, args.optimizer, Bs, show_plot=True, save_plot=False)
                    
                    if all(successes):
                        print("    [PASS] Fit meets all physical criteria!")
                        passed_all_tests = True
                        initial_guess = result.x 
                    else:
                        current_guess[3] *= 1.7
                else:
                    print(f"  V0={current_guess[3]:.2f} | [FAIL] Optimizer did not converge.")
                    current_guess[3] *= 1.7
                    
            if not passed_all_tests:
                print(f"  Giving up on weight {w:.2f}. V0 exceeded bounds.")


        # --- FINAL OUTPUT AND ALWAYS SAVING ---
        print("\n=========================================")
        if best_fit_params is not None:
            A_phys = best_fit_params[0] / Bs
            pass_all = all(best_successes)
            
            print(f"FINAL BEST FIT (Weight = {best_weight:.2f}):")
            print(f"A_phys = {A_phys:.4e}")
            print(f"alpha  = {best_fit_params[1]:.4f}")
            print(f"beta   = {best_fit_params[2]:.4f}")
            print(f"V0     = {best_fit_params[3]:.4f}")
            print(f"gamma  = {best_fit_params[4]:.4f}")
            print(f"Passed Error Checks: {pass_all}")
            
            # 1. Always save the final best plot 
            plot_and_save_fit(best_fit_params, Te_data, target_data_scaled, species, charge_state, density_log10, best_weight, run_id, args.optimizer, Bs, show_plot=False, save_plot=True)
            
            # 2. Append to Database Text File
            db_path = "results/fit_database.txt"
            file_exists = os.path.isfile(db_path)
            
            with open(db_path, "a") as f:
                if not file_exists:
                    f.write("run_id,species,charge,density_log10,optimizer,weight,A,alpha,beta,V0,gamma,max_err_z1,max_err_z2,max_err_z3,passed_all\n")
                
                row_data = [
                    run_id, species, charge_state, str(density_log10), args.optimizer, 
                    f"{best_weight:.2f}", f"{A_phys:e}", f"{best_fit_params[1]:.4f}", 
                    f"{best_fit_params[2]:.4f}", f"{best_fit_params[3]:.4f}", f"{best_fit_params[4]:.4f}",
                    f"{best_max_error[0]:.4f}", f"{best_max_error[1]:.4f}", f"{best_max_error[2]:.4f}", 
                    str(pass_all)
                ]
                f.write(",".join(row_data) + "\n")
            print(f"-> Data appended to {db_path}")

        else:
            print("Optimizer failed to find any valid fits.")


    finally:
        # This block executes unconditionally, cleanly returning the license
        if eng is not None:
            print("Shutting down MATLAB engine and releasing license...")
            eng.quit()