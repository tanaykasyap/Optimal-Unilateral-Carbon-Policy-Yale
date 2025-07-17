# scenario_runner.py, this just defines the scenario runner function.

import pandas as pd
from master import *

def run_scenario(scenario, epsilonSstar1, special_case, data, phi_list, pe_bar, output_root):
    # Your existing function logic here
    theta = 4
    rho = 0.5
    epsbarg = 0.22
    sigma = 0.6
    sigmaE = 1

    if special_case == 'high_theta':
        theta = 16
        rho = 0.5
    elif special_case == 'high_rho':
        rho = 2
        theta = 4

    alphatilde = (pe_bar)**(rho-1) * (epsbarg/(1-epsbarg))
    alpha = alphatilde**(1/rho) / (1+alphatilde**(1/rho))

    if scenario == 'renewable':
        epsilonS1 = epsilonSstar1 = epsilonSstar2 = epsilonS2 = 0.5
        h1, h2 = 1, 0
        epsilonSvec = [(epsilonS1, h1, 0.867), (epsilonS2, h2, 0.133)]
        epsilonSstarvec = [(epsilonSstar1, h1, 0.867), (epsilonSstar2, h2, 0.133)]
    else:
        epsilonS1 = 0.5
        h1 = 1
        epsilonSvec = [(epsilonS1, h1, 1)]
        epsilonSstarvec = [(epsilonSstar1, h1, 1)]

    if special_case == 'none':
        filename = f"{output_root}direct_consumption_{scenario}.csv"
        if epsilonSstar1 == 2 and scenario != 'renewable':
            filename = f"{output_root}direct_consumption_higheps_{scenario}.csv"
    else:
        filename = f"{output_root}direct_consumption_{scenario}_{special_case}.csv"
        if epsilonSstar1 == 2 and scenario != 'renewable':
            filename = f"{output_root}direct_consumption_higheps_{scenario}_{special_case}.csv"

    assert sum(k for i, j, k in epsilonSvec) == 1
    assert sum(k for i, j, k in epsilonSstarvec) == 1

    data_scenario = data.copy()
    if scenario == 'constrained':
        data_scenario = data_scenario[data_scenario['region_scenario'] == 3]
        tax_scenario = ['purete', 'puretc', 'puretp', 'EC_hybrid', 'EP_hybrid', 'PC_hybrid', 'EPC_hybrid']
    elif scenario == 'global':
        data_scenario = data_scenario[data_scenario['region_scenario'] == 3]
        tax_scenario = ['global']
    elif scenario == 'opt':
        data_scenario = data_scenario[data_scenario['region_scenario'] != 4]
        tax_scenario = ['Unilateral']
    elif scenario == 'renewable':
        data_scenario = data_scenario[data_scenario['region_scenario'] == 3]
        tax_scenario = ['Unilateral', 'purete', 'puretc', 'puretp', 'EC_hybrid', 'EP_hybrid', 'PC_hybrid', 'EPC_hybrid']

    model_parameters = (theta, sigma, sigmaE, epsilonSvec, epsilonSstarvec, rho, alpha, pe_bar)
    model = taxModel(data_scenario, tax_scenario, phi_list, model_parameters)
    model.solve()
    model.retrieve(filename)

    #print(f"Results saved to {filename}")
    #print("="*80)  # separator line
    #print(f"Results for scenario {scenario} (Elasticity: {epsilonSstar1}, Special Case: {special_case}) saved to {filename}", flush=True)
    #print("="*80, flush=True)  # separator line for clarity
    output = f"Scenario: {scenario}, Elasticity: {epsilonSstar1}, Special Case: {special_case}\n"
    output += f"Results saved to {filename}\n"
    output += "=" * 80 + "\n"

     
    return output
