import json
import os
from multiprocessing import Pool


import examples.classification
import examples.autoencoder
import examples.regression
import examples.composite
import inspect


def run_module_main_with_args(module_info):
    module_name, fns = module_info
    if "args_fn" in fns and "main_fn" in fns and fns["args_fn"] and fns["main_fn"]:
        print(f"Running {module_name}...")
        args = fns["args_fn"](epochs=1)
        result = fns["main_fn"](args)
        return module_name, result
    else:
        print(f"Skipping {module_name}, missing args_fn or main_fn.")
        return module_name, None


def gather_module_info(packages):
    modules_dict = {}
    for package in packages:
        public_attributes = [attr for attr in dir(package) if not attr.startswith('_')]

        for attr in public_attributes:
            module = getattr(package, attr)
            if inspect.ismodule(module):
                module_key = f"{package.__name__}.{attr}"
                modules_dict[module_key] = {}
                if hasattr(module, 'default_args') and inspect.isfunction(getattr(module, 'default_args')):
                    modules_dict[module_key]['args_fn'] = getattr(module, 'default_args')
                if hasattr(module, 'main') and inspect.isfunction(getattr(module, 'main')):
                    modules_dict[module_key]['main_fn'] = getattr(module, 'main')

    # Filter out modules that don't contain either default_args or main
    modules_dict = {k: v for k, v in modules_dict.items() if 'args_fn' in v and 'main_fn' in v}
    return modules_dict


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method("spawn")
    packages = [
        examples.classification,
        examples.autoencoder,
        examples.regression,
        examples.composite
    ]
    modules_dict = gather_module_info(packages)
    modules_info = list(modules_dict.items())

    for cls, fns in modules_dict.items():
        args = fns["args_fn"](epochs=1)  # Attempt to override epochs
        assert args.epochs == 1, "We should be able to override args"

    modules_info = list(modules_dict.items())

    results_json = {}

    # Setup multiprocessing pool
    with Pool(processes=os.cpu_count()) as pool:
        for module_name, result in pool.imap_unordered(run_module_main_with_args, modules_info):
            if result is not None:
                results_json[module_name] = result
                # Save/update results incrementally
                with open('module_results.json', 'w') as f:
                    json.dump(results_json, f, indent=4)
                print(f"Results saved for {module_name}.")


    # Optionally, save the results to a JSON file
    with open('module_results.json', 'w') as f:
        json.dump(results_json, f, indent=4)

    print("Results saved to module_results.json.")
