from __future__ import print_function
import numpy as np

# Written by Andrew Ning.  Feb 2016.
# FLOW Lab, Brigham Young University.

print('--- starting matlab engine ---')

try:
    import matlab
    import matlab.engine
except ImportError:
    import warnings
    warnings.warn("""
    Matlab engine not installed.
    Instructions here: http://www.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html

    If still having problems, try setting DYLD_FALLBACK_LIBRARY_PATH to contain your python lib location.
    See: http://www.mathworks.com/matlabcentral/answers/233539-error-importing-matlab-engine-into-python
    """)



def fmincon(function, x0, lb, ub, options={}, A=[], b=[], Aeq=[], beq=[],
        providegradients=False, eng=None):

    # check if setpython has a path (e.g., /usr/bin/python)
    # start_options = '-nodesktop -nojvm'
    # if setpython is not None:
    #     start_options += ' -r pyversion ' + setpython

    # start matlab engine
    # eng = matlab.engine.start_matlab(start_options)

    # if eng is None:
    #     import matlab.engine
    eng = matlab.engine.start_matlab()

    # convert to numpy array then list then to matlab type
    # these first conversions are necessary to allow both numpy and list style inputs
    # x0 = matlab.double(np.array(x0).tolist())
    # ub = matlab.double(np.array(ub).tolist())
    # lb = matlab.double(np.array(lb).tolist())

    # A = matlab.double(np.array(A).tolist())
    # b = matlab.double(np.array(b).tolist())
    # Aeq = matlab.double(np.array(Aeq).tolist())
    # beq = matlab.double(np.array(beq).tolist())

    # Cast bounds and constraints to MATLAB doubles
    x0 = matlab.double(np.array(x0).tolist())
    lb = matlab.double(np.array(lb).tolist()) if len(lb) > 0 else matlab.double([])
    ub = matlab.double(np.array(ub).tolist()) if len(ub) > 0 else matlab.double([])
    
    A = matlab.double(A) if len(A) > 0 else matlab.double([])
    b = matlab.double(b) if len(b) > 0 else matlab.double([])
    Aeq = matlab.double(Aeq) if len(Aeq) > 0 else matlab.double([])
    beq = matlab.double(beq) if len(beq) > 0 else matlab.double([])

    # run fmincon
    print('--- calling fmincon ---')
    [xopt, fopt, exitflag, output] = eng.optimize(function, x0, A, b,
        Aeq, beq, lb, ub, options, providegradients, nargout=4)

    xopt = xopt[0]  # convert nX1 matrix to array
    exitflag = int(exitflag)

    # close matlab engine
    eng.quit()

    return xopt, fopt, exitflag, output
