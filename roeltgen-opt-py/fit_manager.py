# main scrip for running the optimization loop and error analysis, mimicking the structure of Roeltgen's MATLAB code
# it runs the optimization for Hydrogen 

import numpy as np
from scipy.integrate import quad

# Import from your other modules
from data_parser import load_adas_plt_h, get_lz_si
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
        val_1, _ = quad(safe_integrand, 0, V0, 
                        args=(Te_j, A_scaled, alpha, beta, V0, gamma),
                        epsabs=1e-8, epsrel=1e-8)
        val_2, _ = quad(safe_integrand, V0, np.inf, 
                        args=(Te_j, A_scaled, alpha, beta, V0, gamma),
                        epsabs=1e-8, epsrel=1e-8)
        calcY[j] = val_1 + val_2
    return calcY

if __name__ == "__main__":
    print("--- Starting Radiation Fitter ---")
    
    # 1. Load Data
    # For Hydrogen 
    interp = load_adas_plt_h("plt96_h.dat")

    Te_data = 10 ** interp.get_knots()[0][3:-3]
    # print("Te Data Points (eV):", Te_data)
    # print(Te_data.size)
    # exit()

    # Te_data = np.geomspace(0.4, 1e3, 15)
    
    Bs = 1e30
    target_data_unscaled = np.array([get_lz_si(1e19, te, interp) for te in Te_data])
    target_data_scaled = target_data_unscaled * Bs
    
    # 2. Setup Loop Parameters
    # Order: [A_scaled, alpha, beta, V0, gamma]
    initial_guess = [0.02, 8e3, 0.8, 3.0, -4.0] 
    #initial_guess = [5.5949e-32*Bs, 8, 7.9587517e-1, 3.52, -1.391] # Initial guess for the 5 parameters
    weight_powers = np.arange(0.1, 0.31, 0.05) 
    print("Weight Powers to Test:", weight_powers)
    best_fit_params = None
    best_weight = None
    global_min_error = np.inf
    
    # 3. OUTER LOOP: Iterate through the weight powers
    for w in weight_powers:
        print(f"\nTesting Weight Power: {w:.2f}")
        
        # Create a working copy of the guess so we can modify it in the inner loop
        current_guess = list(initial_guess)
        print(current_guess)
        # exit()
        passed_all_tests = False
        
        # 4. INNER LOOP: The V0 Kick (Equivalent to MATLAB's while loop)
        while current_guess[3] < 20 and not passed_all_tests:
            print(f"  -> Running optimizer with V0 guess = {current_guess[3]:.2f}")
            
            result = run_single_optimization(current_guess, Te_data, target_data_scaled, w)
            
            if result.success:
                print("Optimizer result:")
                print(result.x)
                calcY_scaled = get_model_emissivity(result.x, Te_data)
                #calcY_scaled = get_model_emissivity(initial_guess, Te_data)
                ratio = np.maximum(calcY_scaled / target_data_scaled, target_data_scaled / calcY_scaled)
                print("Ratio of Model to Target:", ratio)
                # Send to Error Analysis Judge
                successes, max_error = error_analysis(ratio, Te_data, target_data_scaled)
                
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
                    print("    [FAIL] Fit criteria not met. Kicking V0 by 1.25x and retrying...")
                    current_guess[3] *= 1.2 # Increase V0 just like MATLAB
            else:
                print("    [FAIL] Optimizer did not converge. Kicking V0 by 1.25x and retrying...")
                current_guess[3] *= 1.2
                
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