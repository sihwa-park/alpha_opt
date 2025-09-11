import numpy as np
from libensemble.tools.persistent_support import PersistentSupport
from libensemble.message_numbers import EVAL_GEN_TAG, FINISHED_PERSISTENT_GEN_TAG, PERSIS_STOP, STOP_TAG
import nlopt

class QuitOptimizationException(Exception):
    """Custom exception to exit an optimization."""
    pass

def nlopt_gen_f(H, persis_info, gen_specs, libE_info):
    """LibEnsemble generator function that uses NLopt for optimization.
    
    Configuration is passed through gen_specs["user"].
    This function must be at module level (not nested) so it can be pickled
    for multiprocessing.
    """
    # Get configuration from gen_specs
    user_specs = gen_specs["user"]
    algorithm = user_specs["algorithm"]
    x0 = user_specs["x0"]
    initial_step_size = user_specs["initial_step_size"]
    
    print("Entering gen_f. LibE_info:", libE_info)
    ps = PersistentSupport(libE_info, EVAL_GEN_TAG)

    dim_x = len(x0)
    opt = nlopt.opt(algorithm, dim_x)

    def objective(x, grad):
        print("gen_f received x:", x, flush=True)
        batch_size = 1
        H_o = np.zeros(batch_size, dtype=gen_specs["out"])
        H_o["x"][:] = x
        tag, Work, calc_in = ps.send_recv(H_o)
        print("gen_f received tag:", tag, "Work:", Work, "calc_in:", calc_in, flush=True)
        if tag in [STOP_TAG, PERSIS_STOP]:
            raise QuitOptimizationException("Received stop tag from manager")
        return calc_in[0][1]  # Is there a cleaner way to get the objective value?

    opt.set_min_objective(objective)
    opt.set_initial_step(np.full(dim_x, initial_step_size))
    try:
        xopt = opt.optimize(x0)
    except QuitOptimizationException:
        print("gen_f: Optimization stopped by user or manager.", flush=True)

    return None, persis_info, FINISHED_PERSISTENT_GEN_TAG
