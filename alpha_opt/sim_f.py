import numpy as np

def minimal_sim_f(H, persis_info, sim_specs, libE_info):
    """LibEnsemble simulation function that evaluates the objective.
    
    The objective function is passed through sim_specs["user"]["objective"].
    This function must be at module level (not nested) so it can be pickled
    for multiprocessing.
    """
    # Get the objective function from sim_specs
    objective = sim_specs["user"]["objective"]
    
    batch = len(H["x"])  # Num evaluations each sim_f call.
    H_o = np.zeros(batch, dtype=sim_specs["out"])  # Define output array H

    print("sim_f called. x.shape:", H["x"].shape, "x:", H["x"], "libE_info:", libE_info, flush=True)

    for i, x in enumerate(H["x"]):
        f = objective(x)
        H_o["f"][i] = f

    print("sim_f about to return H_o:", H_o, flush=True)
    return H_o, persis_info
