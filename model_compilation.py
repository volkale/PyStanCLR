import pystan
import pickle
import os
import hashlib


def compile_and_pickle_stan_model(stan_model_file):
    """Compiles the Stan model and pickles it for future use."""
    # Read the Stan model code from file
    with open(stan_model_file, 'r') as file:
        stan_code = file.read()

    sha1 = hashlib.sha1()
    sha1.update(str.encode(stan_code))
    seed = int(sha1.hexdigest(), 16) % 4294967295  

    compiled_model = f'lm_model_{seed}.pkl'    
    
    # Compile the model
    sm = pystan.StanModel(model_code=stan_code)

    # Pickle the compiled model
    with open(compiled_model, 'wb') as f:
        pickle.dump(sm, f)
    
    print(f"Stan model compiled and saved to {compiled_model}")
    return sm

def load_compiled_stan_model(stan_model_file):
    """Loads the compiled Stan model from a pickle file."""
    with open(stan_model_file, 'r') as file:
        stan_code = file.read()
    
    sha1 = hashlib.sha1()
    sha1.update(str.encode(stan_code))
    seed = int(sha1.hexdigest(), 16) % 4294967295  

    compiled_model = f'lm_model_{seed}.pkl'    
    
    if not os.path.exists(compiled_model):
        print(f"Compiled model not found. Compiling it now...")
        return compile_and_pickle_stan_model()
    
    # Load the pickled model
    with open(compiled_model, 'rb') as f:
        sm = pickle.load(f)
    
    print(f"Stan model loaded from {compiled_model}")
    return sm
