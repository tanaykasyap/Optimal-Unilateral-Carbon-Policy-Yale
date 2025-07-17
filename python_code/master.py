############################################################################################################################################################
##################################################
########STEP 1 The master file
















import math
import numpy as np
import pandas as pd
from sympy import *
from scipy.integrate import quad
from scipy.optimize import minimize
from scipy.optimize import fsolve
import io
import sys
from multiprocessing import Queue
import io
import sys
import logging
from multiprocessing import Queue
import contextlib  # Add this import


# class object for finding equilibrium taxes
class taxModel:
    def __init__(self, data, tax_scenario, phi_list, model_parameters):
        self.data = data
        self.tax_scenario = tax_scenario
        self.phi_list = phi_list
        self.Qe2 = 0
        self.Qestar2 = 0
        self.theta, self.sigma, self.sigmaE, self.epsilonSvec, self.epsilonSstarvec, self.rho, self.alpha, self.pe_bar = model_parameters
        self.res = []

    # iterate over all combinations of phi, tax scenarios and region scenarios to solve the model
    # pe and tb arguments are initial guesses, default to (1,0)
    #tanay edit changing pe guess to be pe_bar
    def solve(self, pe_guess=166.5, tb_guess=0.0, prop_guess=0.0, te_guess=0.0): #tanay edit changing pe guess to be pe_bar
        for index, region_data in self.data.iterrows():
            for tax in self.tax_scenario:
                pe_prev, tb_prev, prop_prev, te_prev = pe_guess, tb_guess, prop_guess, te_guess
                prices = []
                #print(tax)

                # for each tax-region pair, start with some pe and tb guess
                # then for each value of phi, use the result from the previous iteration as the next guess
                for phi in self.phi_list:
                    

                     
                    # solve model for given phi, tax scenario, region, and guesses for pe, tb, and prop
                    price = self.solveOne(phi, tax, region_data, pe_prev, tb_prev, prop_prev, te_prev)
                     

                    # set new guess to result of previous simulation
                    pe_prev, tb_prev, prop_prev, te_prev, conv = price
                    prices.append((phi, price))
                    print(phi)

                res = pd.Series({'region_data': region_data, 'tax': tax, 'prices': prices})
                self.res.append(res)

    # solve one tax scenario, given phi, tax, region_data and some initial guess of price and taxes
    def solveOne(self, phi, tax, region_data, pe=166.5, tb=0.0, prop=0.0, te=0.0):
        init_guess = [pe, tb, prop]
        if tax == 'global':
            res = self.solve_obj(phi, tax, region_data, init_guess=init_guess)
            opt_val = res[0]

            tb = 0
            te = phi
            prop = 0

        elif tax in ['Unilateral', 'puretc', 'puretp', 'EC_hybrid']:
            res = self.solve_obj(phi, tax, region_data, init_guess=init_guess)
            opt_val = res[0]

            tb = opt_val[1]
            prop = 0
            te = phi

            if tax in ['puretc', 'puretp']:
                te = tb

        elif tax == 'purete':
            res = fsolve(self.te_obj, [pe, te], args=(phi, tax, region_data), full_output=True, maxfev=100000)
            opt_val = res[0]
            te = opt_val[1]


        elif tax == 'PC_hybrid':
            res = self.solve_obj(phi, tax, region_data, init_guess=[pe, tb, prop])
            opt_val = res[0]

            tb = opt_val[1]
            prop = opt_val[2]
            te = tb

        elif tax == 'EP_hybrid':
            res = self.solve_obj(phi, tax, region_data, init_guess=init_guess)
            opt_val = res[0]

            tb = opt_val[1]
            te = opt_val[2]
            prop = te

        elif tax == 'EPC_hybrid':
            # opt_val = self.min_obj(props, tbs, pes, phi, tax, region_data)
            res = self.solve_obj(phi, tax, region_data, init_guess=[pe, tb, prop])
            opt_val = res[0]

            tb = opt_val[1]
            prop = opt_val[2]
            te = phi

        else:
            # tax scenario incorrect
            res = [0, 0, 0]
            opt_val = [0]
            #print('tax scenario', tax, 'not implemented')

        pe = opt_val[0]
        conv = res[2]
        Qe, Qestar, Qes, Qestars = self.comp_qe(tax, pe, tb_mat=[tb, prop], te=te, region_data=region_data)
        if self.phi_list[0] == 0 and len(Qes) > 1:
            self.Qe2 = Qes[1]
        else:
            self.Qe2 = 0

        if self.phi_list[0] == 0 and len(Qestars) > 1:
            self.Qestar2 = Qestars[1]
        else:
            self.Qestar2 = 0
        return pe, tb, prop, te, conv

    # solve system of first order conditions, using some initial guess, with the fsolve function
    #tanay edit changing pe guess to be pe_bar
    def solve_obj(self, phi, tax, region_data, init_guess=[166.5, 0.0, 0.5], verbose=True, second_try=True, tol=1e-10):
        res = fsolve(self.obj_system, init_guess, args=(phi, tax, region_data), full_output=True, maxfev=100000,
                     xtol=tol)
        if res[2] != 1:
            if verbose:
                print("did not converge, tax is", tax, "region is", region_data['regionbase'], 'phi is', phi,
                      'guess is', init_guess)
            if second_try:
                res = fsolve(self.obj_system, [166.5, 0.5*166.5, 0.5], args=(phi, tax, region_data), full_output=True,
                             maxfev=100000) #tanay edit changing pe guess to be pe_bar and also tb guess is scaled by pe_bar
                if res[2] == 1 and verbose:
                    print('converged on second try')
        return res

    # compute system of first order conditions for the case of pure extraction tax
    def te_obj(self, p, phi, tax, region_data):
        p = abs(p)
        pe = p[0]
        te = p[1]
        tb_mat = [0, 1]
        diff, diff1, diff2 = self.comp_obj(pe, te, tb_mat, phi, tax, region_data)

        return diff, diff1

    # compute system of first order conditions. 
    def obj_system(self, p, phi, tax, region_data):
        p = abs(p)
        pe = p[0]
        # combine tb and prop into one vector of tb_mat
        tb_mat = p[1:]
        te = phi

        diff, diff1, diff2 = self.comp_obj(pe, te, tb_mat, phi, tax, region_data)

        return diff, diff1, diff2

    # compute the objective value, currently the objective is to solve the equilibrium conditions
    # (first order condition and world energy market constraint)
    def comp_obj(self, pe, te, tb_mat, phi, tax, region_data):

        # compute extraction tax, and import/export thresholds
        te, tb_mat, j_vals = self.comp_jbar(pe, tb_mat, te, region_data, tax, phi)
        js, jx, jm_hat, jm = j_vals

        # compute energy extraction values    
        Qe, Qestar, Qes, Qestars = self.comp_qe(tax, pe, tb_mat, te, region_data)

        # compute energy consumption values
        cons_vals = self.comp_ce(pe, tb_mat, j_vals, tax, region_data, Qes, Qestars)

        if tax == 'Unilateral':
            sigmatilde = (self.sigma - 1) / self.theta

            # BAU value of home and foreign spending on goods
            Vgx_bar = region_data['Cex_bar'] * self.g(1) / self.gprime(1)

            # counterfactual spending
            pterm = (self.g(pe) / self.g(1)) ** (1 - self.sigma) * Vgx_bar
            num = (1 - js) ** (1 - sigmatilde) - (1 - jx) ** (1 - sigmatilde)
            denum = region_data['jx_bar'] * (1 - region_data['jx_bar']) ** (-sigmatilde)

        # return the objective value
        diff, diff1, diff2 = self.comp_diff(pe, tb_mat, te, phi, Qes, Qestars, Qe, Qestar, j_vals,
                                            cons_vals, region_data, tax)

        return diff, diff1, diff2

    # retrieve results after solving the model, optionally save results as csv
    def retrieve(self, filename=""):
        filled_results = []
        for price_scenario in self.res:
            region_data = price_scenario['region_data']
            tax = price_scenario['tax']
            prices = price_scenario['prices']

            for (phi, (pe, tb, prop, te, conv)) in prices:
                tb_mat = [abs(tb), abs(prop)]
                res = self.comp_all(abs(pe), abs(te), tb_mat, phi, tax, region_data)
                res['regionbase'] = region_data['regionbase']
                res['tax_sce'] = tax
                res['conv'] = conv
                filled_results.append(res)

        # convert the list of pandas series into a pandas dataframe
        df = pd.DataFrame(filled_results)

        # order columns
        cols = list(df.columns.values)
        cols.pop(cols.index('regionbase'))
        cols.pop(cols.index('tax_sce'))
        df = df[['regionbase', 'tax_sce'] + cols]
        df = df[df['conv'] == 1]
        if filename != "":
            df.to_csv(filename, header=True)
            print('file saved to', filename)
        return df

    # returns the difference between world energy consumption and extraction
    def comp_cons_eq(self, pe, te, tb_mat, phi, tax, region_data):
        return self.comp_obj(pe, te, tb_mat, phi, tax, region_data)[0]

    # compute welfare subject to consumption equal to extraction constraint
    def comp_welfare(self, tb_mat, phi, tax, region_data):

        te = phi
        if tax == 'purete':
            te = tb_mat[0]
            tb_mat = [0, 1]

        # find the energy price pe that clears world energy market, given tb at Home
        pe = fsolve(self.comp_cons_eq, [1], args=(te, tb_mat, phi, tax, region_data))[0]

        # compute extraction tax, and import/export thresholds
        te, tb_mat, j_vals = self.comp_jbar(pe, tb_mat, te, region_data, tax, phi)

        # compute extraction values
        Qe_vals = self.comp_qe(tax, pe, tb_mat, te, region_data)
        Qe, Qestar, Qes, Qestars = Qe_vals

        # compute consumption values
        cons_vals = self.comp_ce(pe, tb_mat, j_vals, tax, region_data, Qes, Qestars)

        # compute spending on goods
        vg_vals = self.comp_vg(pe, tb_mat, j_vals, cons_vals, tax, region_data)
        vgfin_vals = self.comp_vgfin(pe, tb_mat, vg_vals, j_vals, tax, region_data)
        Vg_bar, Vg, Vgstar_bar, Vgstar = vgfin_vals

        # compute labour used in goods production
        lg_vals = self.comp_lg(pe, tb_mat, cons_vals, tax, region_data)

        # terms that enter welfare
        delta_vals = self.comp_delta(pe, tb_mat, te, phi, Qes, Qestars, lg_vals, j_vals, vgfin_vals, cons_vals, tax,
                                     region_data)
        delta_Le, delta_Lestar, delta_U, delta_Vg, delta_Vgstar, delta_UCed, delta_UCedstar, delta_emissions, delta_Lg, delta_Lgstar, delta_Le1, delta_Lestar1, delta_Le2, delta_Lestar2, = delta_vals

        # compute welfare change from BAU
        welfare = delta_U

        return welfare

    # compute all values of interest (import/export thresholds, consumption values, welfare etc)
    def comp_all(self, pe, te, tb_mat, phi, tax, region_data):

        # compute extraction tax, and import/export thresholds
        te, tb_mat, j_vals = self.comp_jbar(pe, tb_mat, te, region_data, tax, phi)
        js, jx, jm_hat, jm = j_vals

        # compute energy extraction values
        Qe_vals = self.comp_qe(tax, pe, tb_mat, te, region_data)
        Qe, Qestar, Qes, Qestars = Qe_vals
        Qeworld = Qe + Qestar

        # compute energy consumption values
        cons_vals = self.comp_ce(pe, tb_mat, j_vals, tax, region_data, Qes, Qestars)
        Cey, Cex1, Cex2, Cex, Cem, Ceystar, Ced, Cedstar = cons_vals
        Gestar = Ceystar + Cem + Cedstar
        Cestar = Ceystar + Cex + Cedstar

        # compute spending on goods
        vg_vals = self.comp_vg(pe, tb_mat, j_vals, cons_vals, tax, region_data)
        vgfin_vals = self.comp_vgfin(pe, tb_mat, vg_vals, j_vals, tax, region_data)
        Vg_bar, Vg, Vgstar_bar, Vgstar = vgfin_vals

        subsidy_ratio = tb_mat[0] / pe * self.epsilon_g(pe)

        # compute value of energy used
        ve_vals = self.comp_ve(pe, tb_mat, cons_vals, tax)

        # compute labour used in goods production
        lg_vals = self.comp_lg(pe, tb_mat, cons_vals, tax, region_data)

        # compute leakage values
        leak_vals = self.comp_leak(Qestar, Gestar, Cestar, Qeworld, region_data)

        # terms that enter welfare
        delta_vals = self.comp_delta(pe, tb_mat, te, phi, Qes, Qestars, lg_vals, j_vals, vgfin_vals, cons_vals, tax,
                                     region_data)
        delta_Le, delta_Lestar, delta_U, delta_Vg, delta_Vgstar, delta_UCed, delta_UCedstar, delta_emissions, delta_Lg, delta_Lgstar, delta_Le1, delta_Lestar1, delta_Le2, delta_Lestar2, = delta_vals

        # compute changes from the baseline
        chg_vals = self.comp_chg(Qestar, Gestar, Cestar, Qeworld, region_data)

        # measure welfare and welfare with no emission externality
        welfare = delta_U
        welfare_noexternality = (delta_U + phi * delta_emissions)

        # compute marginal leakage
        leak, leakstar = self.comp_mleak(pe, tb_mat, j_vals, cons_vals, tax)

        # compute Vgx2, spending on exported goods in region 2 for Unilateral policy
        export_subsidy = 0
        if tax == 'Unilateral':
            sigmatilde = (self.sigma - 1) / self.theta

            # BAU value of home and foreign spending on goods
            Vgx_bar = region_data['Cex_bar'] * self.g(1) / self.gprime(1)

            # counterfactual spending
            pterm = (self.g(pe) / self.g(1)) ** (1 - self.sigma) * Vgx_bar
            num = (1 - js) ** (1 - sigmatilde) - (1 - jx) ** (1 - sigmatilde)
            denum = region_data['jx_bar'] * (1 - region_data['jx_bar']) ** (-sigmatilde)
            Vgx2 = pterm * num / denum
            
            export_subsidy = self.g(pe + tb_mat[0]) / self.gprime(pe + tb_mat[0]) * Cex2 - Vgx2

        results = self.assign_val(pe, tb_mat, te, phi, Qeworld, ve_vals, vg_vals, vgfin_vals, delta_vals,
                                  chg_vals, leak_vals, lg_vals, subsidy_ratio, Qe_vals, welfare, welfare_noexternality,
                                  j_vals, cons_vals, leak, leakstar, export_subsidy)

        return results

    # input: pe (price of energy), te (extraction tax), phi (social cost of carbon)
    #        tb_mat (tb_mat[0] = border adjustment,
    #                tb_mat[1] = proportion of tax rebate on exports or extraction tax (in the case of EP_hybrid))
    # output: te (extraction tax)
    #         jx, jm_hat, jm, js (import and export thresholds)
    #         tb_mat (modify tb_mat[1] value to a default value for cases that do not use tb_mat[1])
    def comp_jbar(self, pe, tb_mat, te, region_data, tax, phi):
        # new formulation
        Cey_bar = region_data['Cey_bar']
        Cem_bar = region_data['Cem_bar']
        Cex_bar = region_data['Cex_bar']
        Ceystar_bar = region_data['Ceystar_bar']
        jm_bar = region_data['jm_bar']
        jx_bar = region_data['jx_bar']

        # assign parameters
        theta = self.theta
        g_petb = self.g(pe + tb_mat[0])
        g_pe = self.g(pe)
         # edit- as in table 13 now.
        #jx = g_petb ** (-theta) * Cex_bar / (
               # g_petb ** (-theta) * Cex_bar + (g_pe + tb_mat[0] * self.gprime(pe)) ** (-theta) * Ceystar_bar)
        jx = g_petb ** (-theta) * Cex_bar / (g_petb ** (-theta) * Cex_bar + ((1 +(tb_mat[0]/pe) * self.epsilon_g(pe)) * g_pe) ** (-theta) * Ceystar_bar)       
        js = g_petb ** (-theta) * Cex_bar / (g_petb ** (-theta) * Cex_bar + g_pe ** (-theta) * Ceystar_bar)
        jm_hat = 1

        if tax == 'Unilateral':
            te = phi
            tb_mat[1] = 0

        if tax == 'global':
            te = phi
            tb_mat[0] = 0
            jx = jx_bar
            jm_hat = 1

        if tax == 'purete':
            jx = jx_bar
            jm_hat = 1

        if tax == 'puretc':
            te = tb_mat[0]
            jx = jx_bar
            jm_hat = 1
            tb_mat[1] = 1

        if tax == 'EC_hybrid':
            te = phi
            jx = jx_bar
            jm_hat = 1
            tb_mat[1] = 0

        if tax == 'puretp':
            te = tb_mat[0]
            ve = pe + tb_mat[0]
            g_ve = self.g(ve)
            jm_hat = Cey_bar * (g_pe / g_ve) ** theta / (Cey_bar * (g_pe / g_ve) ** theta + Cem_bar) / jm_bar
            jx = Cex_bar * (g_pe / g_ve) ** theta / (Cex_bar * (g_pe / g_ve) ** theta + Ceystar_bar)
            tb_mat[1] = 0

        if tax == 'EP_hybrid':
            te = tb_mat[1]
            ve = pe + tb_mat[0]
            g_ve = self.g(ve)
            jm_hat = Cey_bar * (g_pe / g_ve) ** theta / (Cey_bar * (g_pe / g_ve) ** theta + Cem_bar) / jm_bar
            jx = Cex_bar * (g_pe / g_ve) ** theta / (Cex_bar * (g_pe / g_ve) ** theta + Ceystar_bar)

        if tax == 'PC_hybrid':
            te = tb_mat[0]
            ve = pe + tb_mat[0] - tb_mat[0] * tb_mat[1]
            g_ve = self.g(ve)
            jm_hat = 1
            jx = Cex_bar * (g_pe / g_ve) ** theta / (Cex_bar * (g_pe / g_ve) ** theta + Ceystar_bar)

        if tax == 'EPC_hybrid':
            te = phi
            ve = pe + tb_mat[0] - tb_mat[0] * tb_mat[1]
            g_ve = self.g(ve)
            jm_hat = 1
            jx = Cex_bar * (g_pe / g_ve) ** theta / (Cex_bar * (g_pe / g_ve) ** theta + Ceystar_bar)

        jm = jm_hat * jm_bar
        j_vals = (js, jx, jm_hat, jm)

        return te, tb_mat, j_vals

    # input: js, jx (export thresholds)
    # output: compute values for the incomplete beta functions, from 0 to js and from 0 to jx
    def incomp_betas(self, js, jx):
        def beta_fun(i, theta, sigma):
            return i ** ((1 + theta) / theta - 1) * (1 - i) ** ((theta - sigma) / theta - 1)

        beta_fun_val1 = quad(beta_fun, 0, js, args=(self.theta, self.sigma))[0]
        beta_fun_val2 = quad(beta_fun, 0, jx, args=(self.theta, self.sigma))[0]
        return beta_fun_val1, beta_fun_val2

    # input: pe (price of energy), tb_mat (tax vector), te (nominal extraction tax), region_data
    # output: home and foreign extraction values, Qes and Qestars are vectors if there are 
    # multiple sources of energy
    def comp_qe(self, tax, pe, tb_mat, te, region_data):
        epsilonSvec, epsilonSstarvec, pe_bar = self.epsilonSvec, self.epsilonSstarvec, self.pe_bar
        #Qe_bar, Qestar_bar = region_data['Qe_bar'], region_data['Qestar_bar']

        Qe1_bar, Qe1star_bar = region_data['Qe1_bar'], region_data['Qe1star_bar']
        Qe2_bar, Qe2star_bar = region_data['Qe2_bar'], region_data['Qe2star_bar']



        Qes = []
        Qe = 0
        # Qe1 is Qc and Qe1star is Qc star. 
        # compute energy extraction for the ith type of energy
        for i in range(len(epsilonSvec)):
            petbte = max((pe + tb_mat[0] - te * epsilonSvec[i][1]) / pe_bar, 0)
            epsS = epsilonSvec[i][0]
            prop = epsilonSvec[i][2]
            #Qe_i = Qe_bar * prop * petbte ** epsS
            Qe_i = Qe1_bar * prop * petbte ** epsS
            Qe += Qe_i #Qe= Qe + Qe_i
            Qes.append(Qe_i)

        Qestars = []
        Qestar = 0
        # compute energy extraction for the ith type of energy
        for i in range(len(epsilonSstarvec)):
            epsSstar = epsilonSstarvec[i][0]
            prop = epsilonSstarvec[i][2]
            #Qestar_i = Qestar_bar * prop * (pe / pe_bar) ** epsSstar
            Qestar_i = Qe1star_bar * prop * (pe / pe_bar) ** epsSstar

            if tax == 'global':
                petbte = max((pe - te * epsilonSstarvec[i][1]) / pe_bar, 0)
                #Qestar_i = Qestar_bar * prop * petbte ** epsSstar
                Qestar_i = Qe1star_bar * prop * petbte ** epsSstar
            Qestar += Qestar_i
            Qestars.append(Qestar_i)

        #print("Qes:", Qes, flush=True)
        #print("Qestars:", Qestars, flush=True)


        return Qe, Qestar, Qes, Qestars

    # input: pe (price of energy), tb_mat (border adjustment and export rebate/extraction tax, depending on tax scenario)
    #        jvals(tuple of jx, jm, js and their hat values (to simplify later computation))
    #        paralist (tuple of user selected parameter)
    #        df, tax_scenario
    # output: detailed energy consumption values (home, import, export, foreign
    #         and some hat values for simplifying calculation in later steps)
    def comp_ce(self, pe, tb_mat, j_vals, tax, region_data, Qes, Qestars):
        js, jx, jm_hat, jm = j_vals
        sigma, theta, sigmaE, pe_bar = self.sigma, self.theta, self.sigmaE, self.pe_bar
        sigmatilde = (sigma - 1) / theta
        Ced_bar, Cedstar_bar = region_data['Ced_bar'], region_data['Cedstar_bar']
        Cey_bar, Cex_bar = region_data['Cey_bar'], region_data['Cex_bar']
        Cem_bar, Ceystar_bar = region_data['Cem_bar'], region_data['Ceystar_bar']
        jm_bar, jx_bar = region_data['jm_bar'], region_data['jx_bar']
        


        print(f"Before adjustments in Comp_Ce -  Cedstar_bar: {Cedstar_bar}", flush=True)
        if len(Qes) > 1 and len(Qestars)>1:
            Ced_bar = region_data['Ced_bar'] + region_data['Qe2_bar'] 
            Cedstar_bar = region_data['Cedstar_bar'] + region_data['Qe2star_bar']
            
        print(f"After adjustments in Comp_Ce -  Cedstar_bar: {Cedstar_bar}", flush=True)
            
           

        # compute incomplete beta values
        beta_fun_val1, beta_fun_val2 = self.incomp_betas(js, jx)

        # direct consumption of energy
        Ced = ((pe + tb_mat[0]) / pe_bar) ** (-sigmaE) * (Ced_bar )
        Cedstar = (pe / pe_bar) ** (-sigmaE) * (Cedstar_bar)

        # Cey
        Cey = self.D(pe + tb_mat[0]) / (self.D(pe_bar)) * Cey_bar

        # Cex1, Cex2 in the Unilaterally optimal case
        Cex1 = self.D(pe + tb_mat[0]) / self.D(pe_bar) * (js / jx_bar) ** (1 - sigmatilde) * Cex_bar

        B1_1 = (1 - sigmatilde) * ((1 - jx_bar) / jx_bar) ** (sigma / theta) * (
                    self.g(pe) / self.g(pe + tb_mat[0])) ** (-sigma)
        B1_2 = self.D(pe + tb_mat[0]) / self.D(pe_bar) * (beta_fun_val2 - beta_fun_val1) / jx_bar ** (1 - sigmatilde)
        B1 = B1_1 * B1_2
        Cex2 = B1 * Cex_bar
        Cex = Cex1 + Cex2

        # Cem, home imports
        Cem = self.D(pe + tb_mat[0]) / self.D(pe_bar) * Cem_bar

        # Ceystar, foreign production for foreign consumption
        Ceystar = self.D(pe) / self.D(pe_bar) * ((1 - jx) / (1 - jx_bar)) ** (1 - sigmatilde) * Ceystar_bar

        # any scenario but Unilateral
        if tax in ['puretc', 'purete', 'EC_hybrid']:
            Cex = self.D(pe) / self.D(pe_bar) * Cex_bar

        elif tax in ['puretp', 'EP_hybrid']:
            tp = tb_mat[0]
            Cey = self.D(pe + tp) / self.D(pe_bar) * (jm / jm_bar) ** (1 - sigmatilde) * Cey_bar
            Cex = self.D(pe + tp) / self.D(pe_bar) * (jx / jx_bar) ** (1 - sigmatilde) * Cex_bar
            Ced = ((pe + tp) / pe_bar) ** (-sigmaE) * Ced_bar
            Cem = self.D(pe) / self.D(pe_bar) * ((1 - jm) / (1 - jm_bar)) ** (1 - sigmatilde) * Cem_bar

        elif tax in ['PC_hybrid', 'EPC_hybrid']:
            tp = tb_mat[0] - tb_mat[1] * tb_mat[0]
            tc= tb_mat[0]
            Cex = self.D(pe + tp) / self.D(pe_bar) * (jx / jx_bar) ** (1 - sigmatilde) * Cex_bar
            Cey = self.D(pe + tc) / self.D(pe_bar) * (jm / jm_bar) ** (1 - sigmatilde) * Cey_bar 
            Ced = ((pe + tc) / pe_bar) ** (-sigmaE) * Ced_bar
            Cem = self.D(pe + tb_mat[0]) / self.D(pe_bar) * Cem_bar
            Ceystar = self.D(pe) / self.D(pe_bar) * ((1 - jx) / (1 - jx_bar)) ** (1 - sigmatilde) * Ceystar_bar

            


       
        return Cey, Cex1, Cex2, Cex, Cem, Ceystar, Ced, Cedstar

    # input: pe (price of energy), tb_mat, jvals (import/export thresholds)
    #        consvals (tuple of energy consumption values), df, tax_scenario, paralist
    # output: value of goods (import, export, domestic, foreign)
    def comp_vg(self, pe, tb_mat, j_vals, cons_vals, tax, region_data):

        return 0, 0, 0, 0, 0, 0
        # unpack values from tuples
         
    # input: pe (price of energy), tb_mat (border adjustments), df, tax_scenario
    #        vg_vals (vector of value of spending on goods), paralist, j_vals (vector of import/export margins)
    # output: Vg_bar, Vg, Vgstar_bar, Vgstar (values of home and foreign total spending on goods)
    #         non prime values returned to simplify later computations
    def comp_vgfin(self, pe, tb_mat, vg_vals, j_vals, tax, region_data):
        # unpack parameters
        Vgy, Vgm, Vgx1, Vgx2, Vgx, Vgystar = vg_vals
        js, jx, jm_hat, jm = j_vals
        sigma, theta, pe_bar = self.sigma, self.theta, self.pe_bar
        Cey_bar, Cem_bar = region_data['Cey_bar'], region_data['Cem_bar']
        Cex_bar, Ceystar_bar = region_data['Cex_bar'], region_data['Ceystar_bar']
        jm_bar, jx_bar = region_data['jm_bar'], region_data['jx_bar']
        sigmatilde = (sigma - 1) / theta
        rho= self.rho 
        pe_bar= self.pe_bar
        # home BAU spending on goods
        Vg_bar = 1 / self.epsilon_g(pe_bar) * pe_bar * (Cey_bar + Cem_bar)
        Vg = (self.g(pe + tb_mat[0]) / self.g(pe_bar)) ** (1 - sigma) * Vg_bar   #tanay edit pe to pe_bar 

        if tax in ['puretp', 'EP_hybrid']:
            # value of home and foreign goods
            ve = pe + tb_mat[0]
            #commented out the old equation-Tanay edit
            #Vg = (jm * (self.g(ve) / self.g(pe_bar)) ** (1 - sigma) * (jm / jm_bar) ** (1 - sigmatilde)
                #  + (1 - jm) * (self.g(pe) / self.g(pe_bar)) ** (1 - sigma) * ((1 - jm) / (1 - jm_bar)) ** (
                 #             1 - sigmatilde)) * Vg_bar   #tanay edit in first line pe to ve, 
            
            #old t1 and t2
            #t1= jm_bar * (self.g(ve)/self.g(pe_bar))**(1-sigma) 
            #t2= (1-jm_bar) * (self.g(pe)/self.g(pe_bar))**(1-sigma)
            t1= jm_bar * (self.g(ve)/self.g(pe_bar))**(1-sigma) * (jm / jm_bar) ** (1 - sigmatilde)
            t2= (1-jm_bar) * (self.g(pe)/self.g(pe_bar))**(1-sigma) * ((1 - jm) / (1 - jm_bar)) ** (1 - sigmatilde)
            Vg= (t1+t2)*Vg_bar
            
             
        # Foreign spending on goods
        Vgstar_bar = 1 / self.epsilon_g(pe_bar) * (Cex_bar + Ceystar_bar)
        ve = pe + tb_mat[0]
        if tax in ['PC_hybrid', 'EPC_hybrid']:
            ve = pe + tb_mat[0] - tb_mat[1] * tb_mat[0]
        #commented out the old equation-Tanay edit

        #Vgstar = (jx * (self.g(ve) / self.g(pe_bar)) ** (1 - sigma) * (jx / jx_bar) ** (1 - sigmatilde)
                  #+ (1 - jx) * (self.g(pe) / self.g(pe_bar)) ** (1 - sigma) * ((1 - jx) / (1 - jx_bar)) ** (
                   #           1 - sigmatilde)) * -Vgstar_bar

        #old t3 and t4 tanay edit
        #t3= jx_bar * (self.g(ve)/self.g(pe_bar))**(1-sigma) 
        #t4= (1-jx_bar) * (self.g(pe)/self.g(pe_bar))**(1-sigma)
        t3 = jx_bar * (self.g(ve) / self.g(pe_bar)) ** (1 - sigma) * (jx / jx_bar) ** (1 - sigmatilde)
        t4 = (1 - jx_bar) * (self.g(pe) / self.g(pe_bar)) ** (1 - sigma) * ((1 - jx) / (1 - jx_bar)) ** (1 - sigmatilde)
        Vgstar= (t3+t4)*Vgstar_bar


        

        return Vg_bar, Vg, Vgstar_bar, Vgstar

    # input: pe (price of energy), tb_mat (border adjustments), tax_scenario
    #        cons_vals (tuple of energy consumption values)
    # output: Ve, Vestar (final values of home and foreign energy consumption)
    def comp_ve(self, pe, tb_mat, cons_vals, tax):
        # unpack parameters
        Cey, Cex1, Cex2, Cex, Cem, Ceystar, Ced, Cedstar = cons_vals

        Ve = (pe + tb_mat[0]) * (Cey + Cem)

        if tax in ['puretp', 'EP_hybrid']:
            Ve = (pe + tb_mat[0]) * Cey + pe * Cem

        if tax in ['PC_hybrid', 'EPC_hybrid']:
            Ve = (pe + tb_mat[0]) * Cey + (pe + tb_mat[0]) * Cem

        Vestar = (pe + tb_mat[0]) * Cex + pe * Ceystar

        if tax == 'Unilateral':
            Vestar = (pe + tb_mat[0]) * Cex1 + pe * (Cex2 + Ceystar)

        if tax in ['puretc', 'EC_hybrid']:
            Vestar = pe * (Cex + Ceystar)

        if tax in ['PC_hybrid', 'EPC_hybrid']:
            Vestar = (pe + tb_mat[0] - tb_mat[1] * tb_mat[0]) * Cex + pe * Ceystar

        return Ve, Vestar

    # input: pe (price of energy), tb_mat (border adjustments), df, tax_scenario, cons_vals (tuple of consumptions values)
    # output: Lg/Lgstar (labour employed in production in home and foreign)
    def comp_lg(self, pe, tb_mat, cons_vals, tax, region_data):
        Cey, Cex1, Cex2, Cex, Cem, Ceystar, Ced, Cedstar = cons_vals
        Cey_bar, Cex_bar = region_data['Cey_bar'], region_data['Cex_bar']
        Cem_bar, Ceystar_bar = region_data['Cem_bar'], region_data['Ceystar_bar']
        pe_bar = self.pe_bar

        # epsilon_g function evaluated at the price of energy faced by users at Home
        epsilon_g_ve = self.epsilon_g(pe + tb_mat[0])
        epsilon_g_bar = self.epsilon_g(pe_bar)
        epsilon_g_pe = self.epsilon_g(pe)

        # labour employed in production in home
        Lg_bar = (1 - epsilon_g_bar) / epsilon_g_bar * pe_bar * (Cey_bar + Cex_bar)
        Lg = (1 - epsilon_g_ve) / epsilon_g_ve * (pe + tb_mat[0]) * (Cey + Cex)

        if tax in ['puretc', 'EC_hybrid']:
            Lg = 1 / self.k(pe + tb_mat[0]) * Cey + 1 / self.k(pe) * Cex

        if tax in ['PC_hybrid', 'EPC_hybrid']:
            Lg = 1 / self.k(pe + tb_mat[0]) * Cey + 1 / self.k(
                pe + tb_mat[0] - tb_mat[1] * tb_mat[0]) * Cex

        # labour employed in foreign production
        Lgstar_bar = 1 / self.k(1) * (Cem_bar + Ceystar_bar)
        Lgstar = 1 / self.k(pe + tb_mat[0]) * Cem + 1 / self.k(pe) * Ceystar

        if tax in ['puretp', 'EP_hybrid']:
            Lgstar = 1 / self.k(pe) * (Cem + Ceystar)

        return Lg_bar, Lgstar_bar, Lg, Lgstar

    def comp_Delta_Lg(self, pe, tb_mat, cons_vals, tax, region_data):
        Cey, Cex1, Cex2, Cex, Cem, Ceystar, Ced, Cedstar = cons_vals
        Cey_bar, Cex_bar = region_data['Cey_bar'], region_data['Cex_bar']
        Cem_bar, Ceystar_bar = region_data['Cem_bar'], region_data['Ceystar_bar']
        pe_bar, rho = self.pe_bar, self.rho

        # energy price faced by users at Home
        petb = pe + tb_mat[0]
        #if tax in ['PC_hybrid', 'EPC_hybrid']:
        #    ve = pe + tb_mat[0] - tb_mat[1] * tb_mat[0]

        # epsilon_g function evaluated at the price of energy faced by users at Home
        epsilon_g_petb = self.epsilon_g(petb)
        epsilon_g_bar = self.epsilon_g(pe_bar)
        epsilon_g_pe = self.epsilon_g(pe)

        Delta_Lgy = ((1 - epsilon_g_petb) / epsilon_g_petb * petb / pe_bar * Cey / Cey_bar
                     - (1 - epsilon_g_bar) / epsilon_g_bar) * pe_bar * Cey_bar
        Delta_Lgx = ((1 - epsilon_g_petb) / epsilon_g_petb * petb / pe_bar * Cex / Cex_bar
                     - (1 - epsilon_g_bar) / epsilon_g_bar) * pe_bar * Cex_bar
        Delta_Lgm = ((1 - epsilon_g_petb) / epsilon_g_petb * petb / pe_bar * Cem / Cem_bar
                     - (1 - epsilon_g_bar) / epsilon_g_bar) * pe_bar * Cem_bar
        Delta_Lgystar = ((1 - epsilon_g_pe) / epsilon_g_pe * pe / pe_bar * Ceystar / Ceystar_bar
                         - (1 - epsilon_g_bar) / epsilon_g_bar) * pe_bar * Ceystar_bar

        Delta_Lg = ((1 - epsilon_g_bar) / epsilon_g_bar) * (petb / pe_bar) ** (rho) * ((Cey + Cex) / (Cey_bar + Cex_bar) - 1) * pe_bar * (Cey_bar + Cex_bar)


        if tax in ['puretc', 'EC_hybrid']:
            Delta_Lgx = ((1 - epsilon_g_pe) / epsilon_g_pe * pe / pe_bar * Cex / Cex_bar
                         - (1 - epsilon_g_bar) / epsilon_g_bar) * pe_bar * Cex_bar

        if tax in ['puretp', 'EP_hybrid']:
            Delta_Lgm = ((1 - epsilon_g_pe) / epsilon_g_pe * (pe / pe_bar) * (Cem / Cem_bar)
                         - (1 - epsilon_g_bar) / epsilon_g_bar) * pe_bar * Cem_bar

        if tax in ['PC_hybrid', 'EPC_hybrid']:
            ve = pe + tb_mat[0] - tb_mat[1] * tb_mat[0]
            epsilon_g_ve = self.epsilon_g(ve)
            Delta_Lgx = ((1 - epsilon_g_ve) / epsilon_g_ve * ve / pe_bar * Cex / Cex_bar # tanay edit numerator should be g(ve)
                     - (1 - epsilon_g_bar) / epsilon_g_bar) * pe_bar * Cex_bar

        Delta_Lg = Delta_Lgy + Delta_Lgx
        Delta_Lgstar = Delta_Lgm + Delta_Lgystar

        return Delta_Lg, Delta_Lgstar

    def comp_Delta_Le_emissions(self, Qes, Qestars, region_data):
        epsilonSvec, epsilonSstarvec = self.epsilonSvec, self.epsilonSstarvec
        #Qe_bar, Qestar_bar = region_data['Qe_bar'], region_data['Qestar_bar']
        Qe1_bar, Qe1star_bar = region_data['Qe1_bar'], region_data['Qe1star_bar']
        Qe2_bar, Qe2star_bar = region_data['Qe2_bar'], region_data['Qe2star_bar']
        pe_bar = self.pe_bar

        # change in labour in home/foreign extraction, total world emissions (different from total world extraction)
        # for the ith energy type
        Delta_Le = 0
        Delta_Lestar = 0
        #included the following variables to calculate the change in emissions
        Delta_Le1=0
        Delta_Lestar1=0
        Delta_Le2=0
        Delta_Lestar2=0
        emissions = 0
        emissions_bar = 0
        '''
        for i in range((len(epsilonSvec))):
            # elasticity of extraction of ith source of extraction
            epsilonS_i = epsilonSvec[i][0]
            epsilonSstar_i = epsilonSstarvec[i][0]

            
            



            # carbon content of ith source of energy
            h_i = epsilonSvec[i][1]
            h_istar = epsilonSstarvec[i][1]

            Qe_i_bar = epsilonSvec[i][2] * Qe_bar
            Qestar_i_bar = epsilonSstarvec[i][2] * Qestar_bar

            
            

            # change in labour used to extract energy
            Delta_Le += epsilonS_i / (epsilonS_i + 1) * (
                        (Qes[i] / Qe_i_bar) ** ((epsilonS_i + 1) / epsilonS_i) - 1) * pe_bar * Qe_i_bar
            Delta_Lestar += epsilonSstar_i / (epsilonSstar_i + 1) * (
                        (Qestars[i] / Qestar_i_bar) ** ((epsilonSstar_i + 1) / epsilonSstar_i) - 1) * pe_bar * Qestar_i_bar
            
            
            
        ''' 
            
         

        epsilonS_1 = epsilonSvec[0][0]
        epsilonSstar_1 = epsilonSstarvec[0][0]
        Delta_Le1 += epsilonS_1 / (epsilonS_1 + 1) * (
                        (Qes[0] / Qe1_bar) ** ((epsilonS_1 + 1) / epsilonS_1) - 1) * pe_bar * Qe1_bar
        Delta_Lestar1 += epsilonSstar_1 / (epsilonSstar_1 + 1) * (
                        (Qestars[0] / Qe1star_bar) ** ((epsilonSstar_1 + 1) / epsilonSstar_1) - 1) * pe_bar * Qe1star_bar
        h_1 = epsilonSvec[0][1]
        h_1star = epsilonSstarvec[0][1]
            
            
        if len(epsilonSvec)>1:
             

            epsilonS_2 = epsilonSvec[1][0]
            epsilonSstar_2 = epsilonSstarvec[1][0]
            Delta_Le2 += epsilonS_2 / (epsilonS_2 + 1) * (
                        (Qes[1] / Qe2_bar) ** ((epsilonS_2 + 1) / epsilonS_2) - 1) * pe_bar * Qe2_bar
            Delta_Lestar2 += epsilonSstar_2 / (epsilonSstar_2 + 1) * (
                        (Qestars[1] / Qe2star_bar) ** ((epsilonSstar_2 + 1) / epsilonSstar_2) - 1) * pe_bar * Qe2star_bar
            h_2 = epsilonSvec[1][1]
            h_2star = epsilonSstarvec[1][1]
                
        Delta_Le = Delta_Le1 + Delta_Le2
        Delta_Lestar = Delta_Lestar1 + Delta_Lestar2
            

        # differs from energy extraction since this is counting carbon emissions
        #emissions += Qes[i] * h_i
        #emissions += Qestars[i] * h_istar
        #emissions_bar += Qe_i_bar * h_i
        #emissions_bar += Qestar_i_bar * h_istar

        emissions += Qes[0] * h_1
        emissions += Qestars[0] * h_1star
        emissions_bar += Qe1_bar * h_1
        emissions_bar += Qe1star_bar * h_1star

        # change in world emissions, same as change in world extraction if only one energy source
        Delta_emissions =  emissions - emissions_bar

        return Delta_Le, Delta_Lestar, Delta_emissions , Delta_Le1, Delta_Lestar1, Delta_Le2, Delta_Lestar2

    def comp_Delta_Vg(self, pe, tb_mat, j_vals, tax, region_data):

        pe_bar, theta, sigma = self.pe_bar, self.theta, self.sigma
        Cey_bar, Cem_bar = region_data['Cey_bar'], region_data['Cem_bar']
        Cex_bar, Ceystar_bar = region_data['Cex_bar'], region_data['Ceystar_bar']
        jx_bar, jm_bar = region_data['jx_bar'], region_data['jm_bar']
        js, jx, jm_hat, jm = j_vals
        sigmatilde = (sigma - 1) / theta

        # epsilon_g function evaluated at various energy prices
        epsilon_g_bar = self.epsilon_g(pe_bar)
        Vg_bar = pe_bar / epsilon_g_bar * (Cey_bar + Cem_bar)
        Vgstar_bar = pe_bar / epsilon_g_bar * (Cex_bar + Ceystar_bar)

        if sigma == 1:
            # values in unilateral optimal, also applies to some of the constrained policies
            Delta_Vg = -math.log(self.g(pe + tb_mat[0]) / self.g(pe_bar)) * Vg_bar
            Delta_Vgstar = -(math.log(self.g(pe) / self.g(pe_bar)) + 1 / theta * math.log(
                (1 - js) / (1 - jx_bar))) * Vgstar_bar

            if tax in ['puretc', 'purete', 'EC_hybrid']:
                Delta_Vgstar = -math.log(self.g(pe) / self.g(pe_bar)) * Vgstar_bar

            if tax in ['puretp', 'EP_hybrid']:
                Delta_Vg = -(math.log(self.g(pe) / self.g(pe_bar)) + 1 / theta * math.log(
                    (1 - jm) / (1 - jm_bar))) * Vg_bar
                Delta_Vgstar = -(math.log(self.g(pe) / self.g(pe_bar)) + 1 / theta * math.log(
                    (1 - jx) / (1 - jx_bar))) * Vgstar_bar

            if tax in ['PC_hybrid', 'EPC_hybrid']:
                Delta_Vg = -(math.log(self.g(pe + tb_mat[0]) / self.g(pe_bar))) * Vg_bar
                Delta_Vgstar = -(math.log(self.g(pe) / self.g(pe_bar)) + 1 / theta * math.log(
                    (1 - jx) / (1 - jx_bar))) * Vgstar_bar
        else:
            Delta_Vg = sigma / (sigma - 1) * ((self.g(pe + tb_mat[0]) / self.g(pe_bar)) ** (1 - sigma) - 1) * Vg_bar
            Delta_Vgstar_23 = (self.g(pe) / self.g(pe_bar)) ** (1 - sigma) * (1 - js) ** (1 - sigmatilde) / (1 - jx_bar) ** (-sigmatilde) # 
            Delta_Vgstar_1 = (self.g(pe + tb_mat[0]) / self.g(pe_bar)) ** (1 - sigma) * js ** (1 - sigmatilde) /  jx_bar ** (-sigmatilde)
            Delta_Vgstar = sigma / (sigma - 1) * (Delta_Vgstar_23 + Delta_Vgstar_1 - 1) * Vgstar_bar

            if tax in ['puretc', 'purete', 'EC_hybrid']:
                Delta_Vgstar = sigma / (sigma - 1) * ((self.g(pe) / self.g(pe_bar)) ** (1-sigma) - 1) * Vgstar_bar

            if tax in ['puretp', 'EP_hybrid']:
                # double check if pe and pe + tb_mat order are reversed
                ve = pe + tb_mat[0]
                # Tanay edit: making deltavg1 and delvg2redundant

                #Delta_Vg_1 = jm * (self.g(pe) / self.g(pe_bar)) ** (1 - sigma) * (jm / jm_bar) ** (1-sigmatilde)
                #Delta_Vg_2 = (1 - jm) * (self.g(ve) / self.g(pe_bar)) ** (1 - sigma) * ((1 - jm) / (1 - jm_bar)) ** (1-sigmatilde)
                #Delta_Vg = sigma / (sigma - 1) * (Delta_Vg_1 + Delta_Vg_2 - 1) * Vg_bar
                 
                # Tanay edit new t1 and t2- adding the extra j/jbar terms 
                #t1= jm_bar * (self.g(ve)/self.g(pe_bar))**(1-sigma) 
                #t2= (1-jm_bar) * (self.g(pe)/self.g(pe_bar))**(1-sigma)
                t1= jm_bar * (self.g(ve)/self.g(pe_bar))**(1-sigma) * (jm / jm_bar) ** (1 - sigmatilde)
                t2= (1-jm_bar) * (self.g(pe)/self.g(pe_bar))**(1-sigma) * ((1 - jm) / (1 - jm_bar)) ** (1 - sigmatilde)
                Delta_Vg= sigma / (sigma - 1) * (t1+t2-1) * Vg_bar
                #added in these lines- tanay edit
                #Delta_Vgstar_3 = (self.g(pe) / self.g(pe_bar)) ** (1 - sigma) * (1 - jx) ** (1 - sigmatilde) / (1 - jx_bar) ** (-sigmatilde) # 
                #Delta_Vgstar_1 = (self.g(pe + tb_mat[0]) / self.g(pe_bar)) ** (1 - sigma) * jx ** (1 - sigmatilde) /  jx_bar ** (-sigmatilde)
                #Delta_Vgstar = sigma / (sigma - 1) * (Delta_Vgstar_3 + Delta_Vgstar_1 - 1) * Vgstar_bar 

                #t3= jx_bar * (self.g(ve)/self.g(pe_bar))**(1-sigma) 
                #t4= (1-jx_bar) * (self.g(pe)/self.g(pe_bar))**(1-sigma)
                t3= jx_bar * (self.g(ve) / self.g(pe_bar)) ** (1 - sigma) * (jx / jx_bar) ** (1 - sigmatilde)
                t4 = (1 - jx_bar) * (self.g(pe) / self.g(pe_bar)) ** (1 - sigma) * ((1 - jx) / (1 - jx_bar)) ** (1 - sigmatilde)
                Delta_Vgstar= sigma / (sigma - 1) * (t3+t4-1) * Vgstar_bar

            if tax in ['PC_hybrid', 'EPC_hybrid']:
                tp = tb_mat[0] - tb_mat[1] * tb_mat[0]
                 
                Delta_Vgstar_23 = (self.g(pe) / self.g(pe_bar)) ** (1 - sigma) * (1 - js) ** (1 - sigmatilde) / (1 - jx_bar) ** (-sigmatilde) # 
                Delta_Vgstar_1 = (self.g(pe + tp) / self.g(pe_bar)) ** (1 - sigma) * js ** (1 - sigmatilde) /  jx_bar ** (-sigmatilde)
                Delta_Vgstar = sigma / (sigma - 1) * (Delta_Vgstar_23 + Delta_Vgstar_1 - 1) * Vgstar_bar    


        return Delta_Vg, Delta_Vgstar

    def comp_Delta_VCed(self, Ced, Cedstar, region_data, Qes, Qestars):
        pe_bar, sigmaE = self.pe_bar, self.sigmaE
        Ced_bar, Cedstar_bar = region_data['Ced_bar'], region_data['Cedstar_bar']

        print(f"Before adjustments - Cedstar: {Cedstar}, Cedstar_bar: {Cedstar_bar}", flush=True)
        print(f"Before adjustments in Comp_Ce -  Cedstar_bar: {Cedstar_bar}", flush=True)
        if len(Qes) > 1 and len(Qestars)>1:
            Ced_bar = region_data['Ced_bar'] + region_data['Qe2_bar'] 
            Cedstar_bar = region_data['Cedstar_bar'] + region_data['Qe2star_bar']
         

        print(f"After adjustments - Cedstar: {Cedstar}, Cedstar_bar: {Cedstar_bar}", flush=True)        
        
        # change in direct consumption of energy that enters welfare change
        if sigmaE != 1:
            Delta_VCed = sigmaE / (sigmaE - 1) * ((Ced / Ced_bar) ** ((sigmaE - 1) / sigmaE) - 1) * pe_bar * Ced_bar
            Delta_VCedstar = sigmaE / (sigmaE - 1) * ((Cedstar / Cedstar_bar) ** ((sigmaE - 1) / sigmaE) - 1) * pe_bar * Cedstar_bar
        else:
            Delta_VCed = math.log(Ced / Ced_bar) * pe_bar * Ced_bar
            Delta_VCedstar = math.log(Cedstar / Cedstar_bar) * pe_bar * Cedstar_bar
            
        

        print(  "From DUCedstar -  Cedstar:", Cedstar, "Cedstarbar:", Cedstar_bar, "DeltaVcedstar:" , Delta_VCedstar, flush=True)
        
        return Delta_VCed, Delta_VCedstar

    # input: pe (price of energy), tb_mat (border adjustments), te (nominal extraction tax), df, tax_scenario, varphi,
    #        paralist, vgfin_vals (total spending by Home and Foreign), j_vals (tuple of import/export margins)
    #        Qeworld, lg_vals (labour in Home and Foreign production)
    # output: compute change in Le/Lestar (labour in home/foreign extraction)
    #         change home utility
    def comp_delta(self, pe, tb_mat, te, phi, Qes, Qestars, lg_vals, j_vals, vgfin_vals, cons_vals, tax, region_data):
        # unpack parameters
        Cey, Cex1, Cex2, Cex, Cem, Ceystar, Ced, Cedstar = cons_vals

        # change in value of energy and emissions
        Delta_Le, Delta_Lestar, Delta_emissions, Delta_Le1, Delta_Lestar1, Delta_Le2, Delta_Lestar2  = self.comp_Delta_Le_emissions(Qes, Qestars, region_data)

        # change in labour used in goods production
        Delta_Lg, Delta_Lgstar = self.comp_Delta_Lg(pe, tb_mat, cons_vals, tax, region_data)

        # change in welfare from direct energy consumption
        Delta_VCed, Delta_VCedstar = self.comp_Delta_VCed(Ced, Cedstar, region_data, Qes, Qestars)

        # change in value of goods production
        Delta_Vg, Delta_Vgstar = self.comp_Delta_Vg(pe, tb_mat, j_vals, tax, region_data)

        # term that is common across all delta_U calculations
        const = -Delta_Le - Delta_Lestar - Delta_Lg - Delta_Lgstar - phi * Delta_emissions

        Delta_U = Delta_Vg + Delta_Vgstar + const + Delta_VCedstar + Delta_VCed

        return Delta_Le, Delta_Lestar, Delta_U, Delta_Vg, Delta_Vgstar, Delta_VCed, Delta_VCedstar, Delta_emissions, Delta_Lg, Delta_Lgstar, Delta_Le1, Delta_Lestar1, Delta_Le2, Delta_Lestar2
        

    # input: Qestar (foregin extraction), Gestar (foreign energy use in production)
    #        Cestar (foregin energy consumption), Qeworld (world extraction), df
    # output: returns average leakage for extraction, production and consumption
    def comp_leak(self, Qestar, Gestar, Cestar, Qeworld, region_data):
        Qestar_bar = region_data['Qe1star_bar']
        Gestar_bar = region_data['Gestar_bar']
        Cestar_bar = region_data['Cestar_bar']
        Qeworld_bar = region_data['Qe1world_bar']
        leakage1 = -(Qestar - Qestar_bar) / (Qeworld - Qeworld_bar)
        leakage2 = -(Gestar - Gestar_bar) / (Qeworld - Qeworld_bar)
        leakage3 = -(Cestar - Cestar_bar) / (Qeworld - Qeworld_bar)

        return leakage1, leakage2, leakage3

    # input: df, Qestar (foreign extraction), Gestar (foreign energy use in production)
    #        Cestar (foregin energy consumption), Qeworld (world extraction)
    # output: compute change in extraction, production and consumption of energy relative to baseline.
    def comp_chg(self, Qestar, Gestar, Cestar, Qeworld, region_data):
        chg_extraction = Qestar - region_data['Qe1star_bar']
        chg_production = Gestar - region_data['Gestar_bar']
        chg_consumption = Cestar - region_data['Cestar_bar']
        chg_Qeworld = Qeworld - region_data['Qe1world_bar']

        return chg_extraction, chg_production, chg_consumption, chg_Qeworld

    # input: pe (price of energy), tb_mat (border adjustments), j_vals (import/export thresholds),
    #        cons_vals (energy consumption values), tax (tax scenario)
    # output: marginal leakage (-(partial Gestar / partial ve) / (partial Ge / partial ve))
    #         for different tax scenarios.
    def comp_mleak(self, pe, tb_mat, j_vals, cons_vals, tax):
        Cey, Cex1, Cex2, Cex, Cem, Ceystar, Ced, Cedstar = cons_vals
        js, jx, jm_hat, jm = j_vals
        theta, sigma = self.theta, self.sigma

        # ve is different for puretp/EP and PC/EPC
        ve = 0
        if tax in ['puretp', 'EP_hybrid']:
            ve = (pe + tb_mat[0])
        if tax in ['PC_hybrid', 'EPC_hybrid']:
            ve = (pe + tb_mat[0] - tb_mat[1] * tb_mat[0])
        # if not a production tax, return 0
        if ve == 0:
            return 0, 0

        ## leakage for PC, EPC taxes
        djxdve = -jx * (1 - jx) * self.gprime(ve) / self.g(ve) * theta
        djmdve = -jm * (1 - jm) * self.gprime(ve) / self.g(ve) * theta

        # consumption of energy in goods production
        dceydve = self.Dprime(ve) / self.D(ve) * Cey + \
                  (1 + (1 - sigma) / theta) * djmdve / jm * Cey
        dcemdve = 1 / (1 - jm) * (1 + (1 - sigma) / theta) * (-djmdve) * Cem
        dcexdve = self.Dprime(ve) / self.D(ve) * Cex + \
                  (1 + (1 - sigma) / theta) * Cex / jx * djxdve
        dceystardve = (1 + (1 - sigma) / theta) * Ceystar * (-djxdve) / (1 - jx)

        # direct consumption
        dceddve = -Ced * self.sigmaE / ve

        # we don't include dcedstardve because it is equal to 0, it does not depend on ve, only on pe.
        leak = -(dceystardve + dcemdve) / (dcexdve + dceydve + dceddve)
        leakstar = -dceystardve / dcexdve

        return leak, leakstar

    def comp_eps(self, Qes, Qe, Qestars, Qestar):
        epsilonSvec, epsilonSstarvec = self.epsilonSvec, self.epsilonSstarvec
        epsilonSstar_num, epsilonSstartilde_num, epsilonSw_num, epsilonSwtilde_num = 0, 0, 0, 0

        for i in range(len(epsilonSstarvec)):
            epsilonSstar_num += epsilonSstarvec[i][0] * Qestars[i]
            epsilonSstartilde_num += epsilonSstarvec[i][0] * epsilonSstarvec[i][1] * Qestars[i]

            epsilonSw_num += epsilonSstarvec[i][0] * Qestars[i] + epsilonSvec[i][0] * Qes[i]
            epsilonSwtilde_num += epsilonSstarvec[i][0] * epsilonSstarvec[i][1] * Qestars[i] + epsilonSvec[i][0] * \
                                  epsilonSvec[i][1] * Qes[i]

        epsilonSstar = epsilonSstar_num / Qestar
        epsilonSstartilde = epsilonSstartilde_num / Qestar
        epsilonSw = epsilonSw_num / (Qestar + Qe)
        epsilonSwtilde = epsilonSwtilde_num / (Qestar + Qe)

        return epsilonSstar, epsilonSstartilde, epsilonSw, epsilonSwtilde

    # input: consval (tuple of consumption values), j_vals (tuple of import/export thresholds),
    #        Ge/Gestar (home/foreign production energy use),
    #        Qe/Qestar (home/foreign/world energy extraction),
    #        Vgx2 (intermediate value for value of home exports),
    #        pe (price of energy), tax_scenario, tb_mat (border adjustments), te (extraction tax)
    #        varphi, df
    # output: objective values
    #         diff (difference between total consumption and extraction)
    #         diff1 & diff3 (equation to compute wedge and border rebate as in table 4 in paper)
    def comp_diff(self, pe, tb_mat, te, phi, Qes, Qestars, Qe, Qestar, j_vals, cons_vals, region_data, tax):
        # unpack parameters
        js, jx, jm_hat, jm = j_vals
        Cey, Cex1, Cex2, Cex, Cem, Ceystar, Ced, Cedstar = cons_vals
        sigma, theta, sigmaE, pe_bar = self.sigma, self.theta, self.sigmaE, self.pe_bar
        sigmatilde = (sigma - 1) / theta
        jx_bar, Cex_bar = region_data['jx_bar'], region_data['Cex_bar']

        # compute world energy consumption extraction
        Ceworld = Cey + Cex + Cem + Ceystar + Ced + Cedstar
        Qeworld = Qe + Qestar

        if tax == 'global':
            return Qeworld - Ceworld, 0, 0

        # compute marginal leakage
        leak, leakstar = self.comp_mleak(pe, tb_mat, j_vals, cons_vals, tax)

        # elasticity of energy supply
        # if only one energy source then tilde and non-tilde are equal
        epsilonSstar, epsilonSstartilde, epsilonSw, epsilonSwtilde = self.comp_eps(Qes, Qe, Qestars, Qestar)

        # first equilibrium condition: world extraction = world consumption
        diff = Qeworld - Ceworld
        # initialize values
        diff1, diff2 = 0, 0

        # evaluate D() function at pe
        Dprime_pe = self.Dprime(pe)
        D_pe = self.D(pe)
        D_petb= self.D(pe + tb_mat[0])
        Dprime_petb = self.Dprime(pe + tb_mat[0])

        if tax == 'Unilateral':
            beta_fun_val1, beta_fun_val2 = self.incomp_betas(js, jx)

            B1_1 = (1 - sigmatilde) * ((1 - jx_bar) / jx_bar) ** (sigma / theta) * (
                    self.g(pe) / self.g(pe + tb_mat[0])) ** (-sigma)
            B1_2 = self.D(pe + tb_mat[0]) / self.D(pe_bar) * (beta_fun_val2 - beta_fun_val1) / jx_bar ** (1 - sigmatilde)
            B1 = B1_1 * B1_2
            B2_1 = (self.g(pe) / self.g(pe_bar)) ** (1-sigma) * ((1 - js) ** (1-sigmatilde) - (1 - jx) ** (1 - sigmatilde))
            B2_2 = pe_bar / self.epsilon_g(pe_bar) / (jx_bar * (1 - jx_bar) ** (-sigmatilde))
            B2 = B2_1 * B2_2
            S = ((pe + tb_mat[0]) / self.epsilon_g(pe + tb_mat[0]) * B1 - B2) * Cex_bar

            epsilon_D_pe = - pe * Dprime_pe / D_pe

            numerator = phi * epsilonSstartilde * Qestar - sigma * self.epsilon_g(pe) * S
            denominator = epsilonSstar * Qestar + epsilon_D_pe * Ceystar + sigmaE * Cedstar
            # border adjustment = consumption wedge
            diff1 = tb_mat[0] * denominator - numerator

        if tax == 'purete':
            dcewdpe = abs(Dprime_pe / D_pe * Cey
                          + Dprime_pe / D_pe * Cex
                          + Dprime_pe / D_pe * (Ceystar + Cem)
                          + -sigmaE * Ced / pe + -sigmaE * Cedstar / pe)

            numerator = phi * epsilonSstartilde * Qestar
            denominator = epsilonSstar * Qestar + dcewdpe * pe

            # te = varphi - consumption wedge
            diff1 = (phi - te) * denominator - numerator

        if tax in ['puretc', 'EC_hybrid']:
            dcestardpe = abs(Dprime_pe / D_pe * Cex
                             + Dprime_pe / D_pe * Ceystar
                             + -sigmaE * Cedstar / pe)

            numerator = phi * epsilonSwtilde * Qeworld
            denominator = epsilonSw * Qeworld + dcestardpe * pe
            if tax == 'EC_hybrid':
                numerator = phi * epsilonSstartilde * Qestar
                denominator = epsilonSstar * Qestar + dcestardpe * pe

            # border adjustment = consumption wedge
            diff1 = tb_mat[0] * denominator - numerator

        if tax in ['puretp', 'EP_hybrid']:
            djxdpe = theta * self.gprime(pe) / self.g(pe) * jx * (1 - jx)
            djmdpe = theta * self.gprime(pe) / self.g(pe) * jm * (1 - jm)
            dceystardpe = (Dprime_pe / D_pe
                           - (1 + (1 - sigma) / theta) / (1 - jx) * djxdpe) * Ceystar
            dcexdpe = (1 + (1 - sigma) / theta) / jx * djxdpe * Cex
            dcemdpe = (Dprime_pe / D_pe
                       - (1 + (1 - sigma) / theta) / (1 - jm) * djmdpe) * Cem
            dceydpe = ((1 + (1 - sigma) / theta) / jm * djmdpe) * Cey
            dcedstardpe = -sigmaE * Cedstar / pe

            numerator = phi * epsilonSwtilde * Qeworld
            denominator = (epsilonSw * Qeworld - (dceystardpe + dcemdpe + dcedstardpe) * pe
                           - leak * (dcexdpe + dceydpe) * pe)
            if tax == 'EP_hybrid':
                numerator = phi * epsilonSstartilde * Qestar
                denominator = (epsilonSstar * Qestar - (dceystardpe + dcemdpe + dcedstardpe) * pe
                               - leak * (dcexdpe + dceydpe) * pe)
                diff2 = (phi - tb_mat[1]) * denominator - leak * numerator

            # border adjustment = (1-leakage) consumption wedge
            diff1 = tb_mat[0] * denominator - (1 - leak) * numerator

        if tax in ['PC_hybrid', 'EPC_hybrid']:
            djxdpe = theta * self.gprime(pe) / self.g(pe) * jx * (1 - jx)
            dceystardpe = (Dprime_pe / D_pe
                           - (1 + (1 - sigma) / theta) / (1 - jx) * djxdpe) * Ceystar
            dcexdpe = ((1 + (1 - sigma) / theta) / jx * djxdpe) * Cex
            dcedstardpe = -sigmaE * Cedstar / pe

            dcezstardpe = dceystardpe + dcedstardpe #Tthis is deceytildestar in the paper 

            numerator = phi * epsilonSwtilde * Qeworld
            denominator = epsilonSw * Qeworld - dcezstardpe * pe - leakstar * dcexdpe * pe
            if tax == 'EPC_hybrid':
                numerator = phi * epsilonSstartilde * Qestar
                denominator = epsilonSstar * Qestar - dcezstardpe * pe - leakstar * dcexdpe * pe

            diff1 = tb_mat[0] * denominator - numerator
            # border rebate for exports tb[1] * tb[0] = leakage * tc
            # in diff2 we should have (1-leakstar)*numerator instead of leakstar
            #also note in diff2 the first element should be tp, which per paper and in the 
            #ce calculuation we defined as tb[0]- tb[1]*tb[0], here we just had tb[1]*tb[0]= tp so made the change.
            diff2 = (tb_mat[0]- tb_mat[1] * tb_mat[0]) * denominator - (1-leakstar) * numerator

        return diff * 100, diff1, diff2

    # assign values to return later
    def assign_val(self, pe, tb_mat, te, phi, Qeworld, ve_vals, vg_vals, vgfin_vals, delta_vals, chg_vals,
                   leak_vals, lg_vals, subsidy_ratio, Qe_vals, welfare, welfare_noexternality, j_vals, cons_vals, leak,
                   leakstar, export_subsidy):
        js, jx, jm_hat, jm = j_vals
        Cey, Cex1, Cex2, Cex, Cem, Ceystar, Ced, Cedstar = cons_vals
        Vgy, Vgm, Vgx1, Vgx2, Vgx, Vgystar = vg_vals
        Vg_bar, Vg, Vgstar_bar, Vgstar = vgfin_vals
        Lg_bar, Lgstar_bar, Lg, Lgstar = lg_vals
        leakage1, leakage2, leakage3 = leak_vals
        delta_Le, delta_Lestar, delta_U, delta_Vg, delta_Vgstar, delta_UCed, delta_UCedstar, delta_emissions, delta_Lg, delta_Lgstar, delta_Le1, delta_Lestar1, delta_Le2, delta_Lestar2 = delta_vals
        Ve, Vestar = ve_vals
        Qe, Qestar, Qes, Qestars = Qe_vals
        chg_extraction, chg_production, chg_consumption, chg_Qeworld = chg_vals
        epsilon_g_check= self.epsilon_g(self.pe_bar)
 
        ret = pd.Series(
            {'varphi': phi, 'pe': pe, 'tb': tb_mat[0], 'petb': pe + tb_mat[0], 'prop': tb_mat[1],  'tp': tb_mat[0] - tb_mat[1]*tb_mat[0], 'te': te, 'jx': jx,
             'jm': jm, 'js': js, 'Qe': Qe, 'Qestar': Qestar, 'Qeworld': Qeworld,
             'Ced': Ced, 'Cedstar': Cedstar, 'Cey': Cey, 'Cex': Cex, 'Cem': Cem, 'Cex1': Cex1, 'Cex2': Cex2,
             'Ceystar': Ceystar, 'Vgm': Vgm, 'Vgx1': Vgx1, 'Vgx2': Vgx2, 'Vgx': Vgx, 'Vg': Vg,
             'Vgstar': Vgstar, 'delta_Lg': delta_Lg, 'delta_Lgstar': delta_Lgstar, 'Ve': Ve, 'Vestar': Vestar, 'delta_Le': delta_Le,
             'delta_Lestar': delta_Lestar, 'delta_Le1': delta_Le1, 'delta_Lestar1': delta_Lestar1, 'delta_Le2': delta_Le2, 'delta_Lestar2': delta_Lestar2,
             'leakage1': leakage1, 'leakage2': leakage2, 'leakage3': leakage3,
             'chg_extraction': chg_extraction, 'chg_production': chg_production,
             'chg_consumption': chg_consumption, 'chg_Qeworld': chg_Qeworld, 'subsidy_ratio': subsidy_ratio,
             'delta_Vg': delta_Vg, 'delta_Vgstar': delta_Vgstar, 'delta_UCed': delta_UCed,
             'delta_UCedstar': delta_UCedstar, 'leak': leak, 'leakstar': leakstar,
             'welfare': welfare, 'welfare_noexternality': welfare_noexternality, 'export_subsidy': export_subsidy, 'epsbarg': epsilon_g_check})
        for i in range(len(Qes)):
            Qe = 'Qe' + str(i + 1)
            Qestar = 'Qe' + str(i + 1) + 'star'
            ret[Qe] = Qes[i]
            ret[Qestar] = Qestars[i]
        return ret

    # define CES production function and its derivative
    def g(self, p):
    
        alpha= self.alpha 
        rho=self.rho
        alphatilde = (self.alpha/(1-self.alpha))** rho 
        if rho == 1:

            return alpha** (-alpha) * (1-alpha)** (alpha-1) * p** self.alpha
        else:
            t1 = (1 - alpha) ** (rho / (1 - rho))
            t2 = (1+ alphatilde * p **(1-rho))   
            t3=  t2 ** (1/(1-rho))
            return t1 * t3

    def gprime(self, p):
        alpha=self.alpha
        rho=self.rho
        alphatilde =(self.alpha/(1-self.alpha))** rho 
        if rho == 1:
            num = alpha
            den = p * (1 - alpha)
            return (num / den)**(1-self.alpha)
        else:
            t1 = (1 - alpha) ** (rho / (1 - rho))
            t2 = (1+ alphatilde * p **(1-rho))**(rho/(1-rho))
            t3 = alphatilde * p** (-rho)
            return t1 * t2 * t3   
        




    def k(self, p):
        return self.gprime(p) / (self.g(p) - p * self.gprime(p))

    # D(p, sigmastar) corresponds to D(p) in paper
    def D(self, p):
        return self.gprime(p) * self.g(p) ** (-self.sigma)

    def Dprime(self, p):
        x = symbols('x')
        return diff(self.D(x), x).subs(x, p)

    def epsilon_g(self, p):
        return p * self.gprime(p) / self.g(p)
    
 
def run_scenario(scenario, epsilonSstar1, special_case, data, phi_list, pe_bar, output_root):
    # Create a StringIO buffer to capture stdout
    buffer = io.StringIO()
    
    # Redirect stdout to the buffer
    with contextlib.redirect_stdout(buffer):
        # Existing run_scenario code
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
            epsilonS1 = epsilonSstar1 = 0.5
            epsilonSstar2 = epsilonS2 = 0.5
            h1, h2 = 1, 0
            epsilonSvec = [(epsilonS1, h1, 1), (epsilonS2, h2, 0.133/0.867)]
            epsilonSstarvec = [(epsilonSstar1, h1, 1), (epsilonSstar2, h2, 0.133/0.867)]
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

        # Scenario-specific data manipulation
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
            tax_scenario = ['Unilateral']

        # Print the initiation message
        print(f"Running scenario: {scenario}, with epsilonSstar1: {epsilonSstar1}, and special_case: {special_case}", flush=True)

        # Initialize and solve the model
        model_parameters = (theta, sigma, sigmaE, epsilonSvec, epsilonSstarvec, rho, alpha, pe_bar)
        model = taxModel(data_scenario, tax_scenario, phi_list, model_parameters)
        model.solve()
        model.retrieve(filename)

        # Prepare the output message
        output = f"Scenario: {scenario}, Elasticity: {epsilonSstar1}, Special Case: {special_case}\n"
        output += f"Results saved to {filename}\n"
        output += "=" * 80 + "\n"

    # Get the captured stdout
    captured_output = buffer.getvalue()

    # Combine the captured stdout with the output message
    final_output = captured_output + output

    return final_output
 

'''
            
def run_scenario(scenario, epsilonSstar1, special_case, data, phi_list, pe_bar, output_root): #Qes, Qestars, Ced_bar, Cedstar_bar):
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
        epsilonS1 = epsilonSstar1 = 0.5
        epsilonSstar2 = epsilonS2 = 2
        h1, h2 = 1, 0
        epsilonSvec = [(epsilonS1, h1, 1), (epsilonS2, h2, 0.133/0.867)]
        epsilonSstarvec = [(epsilonSstar1, h1, 1), (epsilonSstar2, h2, 0.133/0.867)]
        #Ced_bar= data['Ced_bar'] + Qes[1][0]
        #Cedstar_bar= data['Cedstar_bar'] + Qestars[1][0]

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

    #assert sum(k for i, j, k in epsilonSvec) == 1
    #assert sum(k for i, j, k in epsilonSstarvec) == 1

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
        tax_scenario = ['Unilateral']
 

    print(f"Running scenario: {scenario}, with epsilonSstar1: {epsilonSstar1}, and special_case: {special_case}", flush=True)
    
    
    
    
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



    




'''




############################################################################################################################################################
################################################################################################################################################################
## STEP 2 loading in the sub files- Only comment this out when using the cluster, otherwise proceed to running 
## sim. py or exp.ipynb 

#exec(open("code/sim.py").read())
    

    
    