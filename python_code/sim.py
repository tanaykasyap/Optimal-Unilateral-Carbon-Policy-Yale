 

import math
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
#from scenario_runner import run_scenario  # Import the function
from master import *
import warnings
warnings.filterwarnings('ignore')



################################################

if __name__ == "__main__":
    scenarios = [
        ('constrained', 2, 'none'),  # high eps constrained, no special case
        ('constrained', 0.5, 'none'),  # low eps constrained, no special case
        ('constrained', 2, 'high_rho'),  # high eps constrained, high_rho
        ('constrained', 0.5, 'high_rho'),  # low eps constrained, high_rho
        ('constrained', 2, 'high_theta'),  # high eps constrained, high_theta
        ('constrained', 0.5, 'high_theta'),  # low eps constrained, high_theta

        ('opt', 2, 'none'),  # high eps opt, no special case
        ('opt', 0.5, 'none'),  # low eps opt, no special case
        ('opt', 2, 'high_rho'),  # high eps opt, high_rho
        ('opt', 0.5, 'high_rho'),  # low eps opt, high_rho
        ('opt', 2, 'high_theta'),  # high eps opt, high_theta
        ('opt', 0.5, 'high_theta'),  # low eps opt, high_theta

        ('global', 0.5, 'none'),  # global, no special case
        ('global', 0.5, 'high_rho'),  # global, high_rho
        ('global', 0.5, 'high_theta'),  # global, high_theta

        ('renewable', 0.5, 'none'),  # renewable, no special case
        ('renewable', 0.5, 'high_rho'),  # renewable, high_rho
        ('renewable', 0.5, 'high_theta')  # renewable, high_theta
    ]


    lit_param = False
    pe_bar = 166.5
    phi_upper = 3000
    output_root = 'output_phi3000/'

    data = pd.read_excel("data/baselinecarbon_renewables_direct_2018.xlsx")
    data['jx_bar'] = data['Cex_bar'] / (data['Cex_bar'] + data['Ceystar_bar'])
    data['jm_bar'] = data['Cey_bar'] / (data['Cey_bar'] + data['Cem_bar'])

    phi_list = np.arange(0, phi_upper, 0.01) * pe_bar
    phi_list = np.sort(np.append(phi_list, 190))

    all_outputs = []

    with ProcessPoolExecutor(max_workers=18) as executor:
        futures = [
            executor.submit(run_scenario, scenario, epsilonSstar1, special_case, data, phi_list, pe_bar, output_root)
            for scenario, epsilonSstar1, special_case in scenarios
        ]

        for future in futures:
            future.result()
            output = future.result()
            all_outputs.append(output)

    # Print all outputs after all futures have completed
    for output in all_outputs:
        print(output)