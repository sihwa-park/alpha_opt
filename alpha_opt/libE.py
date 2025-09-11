from libensemble.alloc_funcs.start_only_persistent import only_persistent_gens
from libensemble import Ensemble
from libensemble.tools import parse_args

def libE_driver_serial(sim_f, sim_user_specs, gen_f, gen_user_specs, dim_x, sim_max):
    # libensemble.gen_funcs.rc.aposmm_optimizers = "scipy"

    nworkers, is_manager, libE_specs, _ = parse_args()
    print("nworkers:", nworkers)
    print("is_manager:", is_manager)
    print("libE_specs:", libE_specs)
    print("dim_x:", dim_x)

    # https://libensemble.readthedocs.io/en/main/data_structures/libE_specs.html
    libE_specs["sim_dirs_make"] = True  # Create directories for simulations
    # libE_specs["sim_dir_copy_files"] = ["../w7x.json"]

    sim_specs = {
        "sim_f": sim_f,  # Simulation function
        "in": ["x"],  # Accepts 'x' values
        "out": [("f", float)],  # Returns f(x) values
        "user": sim_user_specs,  # User-defined parameters
    }

    # dim_x = 7 # Dimension of the input 'x'
    gen_out = [
        ("x", float, dim_x),  # Produces 'x' values
        # ("x_on_cube", float, dim_x),  # 'x' values scaled to unit cube
        # ("sim_id", int),  # Produces IDs for sim order
        # ("lambda", float),  # dimensionless scaling parameter
        # ("local_min", bool),  # Is a point a local minimum?
        # ("local_pt", bool),  # Is a point from a local opt run?
    ]

    gen_specs = {
        "gen_f": gen_f,
        # "persis_in": ["x", "f", "x_on_cube", "sim_id", "local_min", "local_pt"],
        # "persis_in": ["x", "f", "sim_id", "lambda"],
        # "persis_in": ["x", "f", "lambda"],
        "persis_in": ["x", "f"],
        "out": gen_out,  # Output defined like above dict
        "user": gen_user_specs,  # Merge user-defined parameters
    }

    alloc_specs = {"alloc_f": only_persistent_gens}

    exit_criteria = {"sim_max": sim_max}
    # persis_info = add_unique_random_streams({}, nworkers + 1)

    # H0_filename = "../20250627-01-004_test_uniform_sampling_then_persistent_localopt_runs/uniform_sampling_then_persistent_localopt_runs_history_length=1003_evals=1000_workers=4.npy"
    # H0 = np.load(H0_filename)

    # H, persis_info, flag = libE(sim_specs, gen_specs, exit_criteria, persis_info, alloc_specs, libE_specs, H0=H0)
    # H, persis_info, flag = libE(sim_specs, gen_specs, exit_criteria,
    # persis_info, alloc_specs, libE_specs)
    # ensemble = Ensemble(sim_specs, gen_specs, exit_criteria, libE_specs, alloc_specs, persis_info)
    ensemble = Ensemble(sim_specs, gen_specs, exit_criteria, libE_specs, alloc_specs)
    ensemble.run()

    H = ensemble.H  # Get the history array
    if is_manager:
        ensemble.save_output("results")
        # print("Minima:", H[np.where(H["local_min"])]["x"])