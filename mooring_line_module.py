# -*- coding: utf-8 -*-
"""
Created on Wed May 24 15:47:33 2023

VISIMOOR CODE

contents:
    - Mooring system class

@author: ogf1n20
"""

#%% Init Mooring system class 

import math
import sympy as sp
from scipy import optimize
import matplotlib.pyplot as plt
import numpy as np
import cProfile
import pandas as pd
import timeit

class mooring_line:
    def __init__(self, x_f, z_f, l1, ea1, lrd_type, mooring_type, w1, 
                 HF0=None, VF0=None, profile_res = 40, offset_res = 1,
                 max_offset = 10,
                 debugging=False, *args, **kwargs):
        
        # Physical property inputs
        self.x_f = x_f  # fairlead x position (m)
        self.z_f = z_f  # fairlead z position (m)
        self.l1 = l1  # unstretched length of the mooring line (m)
        self.ea1 = ea1  # Extensional stiffness of the line (N)
        self.mooring_type = mooring_type  # Type of mooring, which can be either taut or catenary
        self.w1 = w1  # weight of line in fluid (N)
        self.HF0 = HF0  # starting guess of the horizontal force, just a number (N)
        self.VF0 = VF0  # starting guess of the vertical force, just a number (N)
        
        # General LRD inputs for both technologies
        self.lrd_type = lrd_type  # type of LRD, which can be either TFI, DO
        # Length of the spring for TFI, or length of the cylinder for DO
        self.lrd_length = kwargs.get('lrd_length') 

        # Handling additional arguments depending on the LRD_type
        if self.lrd_type == "DO":
            self.lrd_o = kwargs.get('lrd_o')  # Additional parameter for DO lrd_type
            self.lrd_v = kwargs.get('lrd_v')  # Additional parameter for DO lrd_type
            self.lrd_fg = kwargs.get('lrd_fg')  # Additional parameter for DO lrd_type

        elif self.lrd_type == "TFI":
            self.rated_strain = kwargs.get('rated_strain') 
            self.rated_tension = kwargs.get('rated_tension')   
           
        # Other inputs
        self.profile_res = profile_res # resolution of plot
        self.offset_res = offset_res # resolution of plot
        self.debugging = debugging # plots and prints intermediate stuff 
        self.max_offset = max_offset 
                
        # Total mooring line length
        self.ltot = self.l1 + self.lrd_length
        
        # Calculating catenary paramater lambda and HF0 VF0 guesses:       
        
        # Calculate lambda
        if math.sqrt(x_f**2 + z_f**2) >= l1:
            lambda_val = 0.2
        else:
            lambda_val = math.sqrt(3 * ((l1**2 - z_f**2) / x_f**2 - 1))

        # Calculate HF0 and VF0 if they are not provided
        if HF0 is None:
            self.HF0 = abs((w1 * x_f) / (2 * lambda_val))  # starting guess of the horizontal force
        else:
            self.HF0 = HF0

        if VF0 is None:
            self.VF0 = 0.5 * w1 * (z_f / math.tanh(lambda_val) + l1)  # starting guess of the vertical force
        else:
            self.VF0 = VF0
        
        if self.debugging == True:
            
            print('Hf0 = ' + str(self.HF0))
            print('Vf0 = ' + str(self.VF0))
        
    def mooring_equations(self): # Function to calculate mooring system equations based on mooring_type
    
        # creating symbolic variables        
        self.Vf, self.Hf = sp.symbols('Vf Hf')
        x_f_sym, z_f_sym = sp.symbols('x_f z_f')  
        self.s = sp.symbols('s') # position along mooring line
    
        # Simplify the expressions a bit
        Vf_Hf = self.Vf / self.Hf  # ratio of vert to horz forces
        tension = sp.sqrt( self.Vf ** 2 + self.Hf ** 2 )
        Lb    = self.l1 - self.Vf/self.w1 # length of chain along seabed for cat
        sqrt_term = sp.sqrt(1 + Vf_Hf**2)
        log_term = sp.log(Vf_Hf + sqrt_term)
        sqrt_term_chain_profile = sp.sqrt(1 + ((self.w1*(self.s - Lb) / self.Hf))**2)
        log_term_chain_profile = sp.log((self.w1*(self.s - Lb) / self.Hf) 
                                        + sqrt_term_chain_profile)    
                    
        # Define extra term based on LRD_type
        if self.lrd_type == "DO":
            LRD_x = 0
            LRD_z = 0
            
        elif self.lrd_type == "TFI":
            
            A,B,C,D,J = 0.0123801, 211.947, 0.751564, 11.4939, -16715.3 # fixed constants
            rt = 485.82 / self.rated_tension
            rs = self.rated_strain / 3.5885 
            
            term1 = (A * ( rt * tension - J) - B) / (1 + (A * (rt * tension - J) - B - C) ** 2)
            term2 = (A * J + B) / (1 + (- A * J - B - C) ** 2)
            term3 = D * sp.sqrt(A * (rt * tension - J))
            term4 = - D * sp.sqrt(-1 * A * J)
            
            strain = rs * (term1 + term2 + term3 + term4)
            
            LRD_x = (self.Hf * self.lrd_length / tension) * (1 + strain)
            LRD_z = (self.Vf * self.lrd_length / tension) * (1 + strain)
            
        else:
            LRD_x = 0
            LRD_z = 0
        
        
        
        if self.mooring_type == 'Catenary':
            
            x_f_eq = (self.l1 - self.Vf/self.w1 + (self.Hf/self.w1) * log_term + self.Hf*self.l1/self.ea1 + LRD_x) - x_f_sym
            z_f_eq = ( (self.Hf/self.w1) * (sqrt_term - 1) + self.Vf**2 / (2 * self.ea1 * self.w1) + LRD_z) - z_f_sym
            
            x_s_eq = sp.Piecewise((self.s, 
                                   self.s <= Lb), # nodes on seabed
                                  (Lb + (self.Hf/self.w1) * log_term_chain_profile +
                                   (self.Hf * self.s) / self.ea1,
                                   sp.And(Lb < self.s, self.s <= self.l1)), # nodes between seabed and LRD
                                   (Lb + (self.Hf/self.w1) * log_term_chain_profile +
                                   (self.Hf * self.s) / self.ea1 + 
                                   LRD_x * (self.s - self.l1) / self.lrd_length, 
                                   sp.And(self.l1 < self.s, self.s <= self.ltot))) # nodes between LRD and fairlead

            
            z_s_eq = sp.Piecewise((0, 
                                   self.s <= Lb), # nodes on seabed
                                  ((self.Hf/self.w1) * (sqrt_term_chain_profile - 1) +
                                   (self.w1*(self.s - Lb)**2) / (2 * self.ea1), 
                                   sp.And(Lb < self.s, self.s <= self.l1)), # nodes between seabed and LRD
                                  ((self.Hf/self.w1) * (sqrt_term_chain_profile - 1) +
                                   (self.w1*(self.s - Lb)**2) / (2 * self.ea1) + 
                                   LRD_z * (self.s - self.l1) / self.lrd_length,  
                                   sp.And(self.l1 < self.s, self.s <= self.ltot))) # nodes between LRD and fairlead
            
        elif self.mooring_type == 'Taut':
            x_f_eq = self.Hf/self.w * (log_term - sp.log((self.Vf - self.w*self.L)/self.Hf + sqrt_term)) + LRD_x - self.x_f
            z_f_eq = ( (self.Hf/self.w) * (sqrt_term - sp.sqrt(1 + ((self.Vf - self.w*self.L) / self.Hf)**2)) + 
                1/self.EA * (self.Vf*self.L - 0.5*self.w*self.L**2) + LRD_z) - self.z_f
            
            x_s_eq = sp.Piecewise((self.s, self.s <= Lb ),
                                  (Lb + (self.Hf/self.w) * log_term_chain_profile 
                                   + (self.Hf * self.s) / self.EA, True))
            
            z_s_eq = sp.Piecewise((0, self.s <= self.L - self.Vf/self.w),
                                  ((self.Hf/self.w) * (sqrt_term_chain_profile - 1) +
                                   (self.w*(self.s - Lb)**2) / (2 * self.EA), True))       
        return x_f_eq, z_f_eq, x_s_eq, z_s_eq, LRD_x, LRD_z
        
    def solve_system(self, x_f_eq, z_f_eq, x_s_eq, z_s_eq, LRD_x, LRD_z, 
                     initial_guess = None):
        
        # STEP 1: get fairlead forces ----------------------------------------
        
        # Get the system of fairlead position functions
        #x_f_eq, z_f_eq, x_s_eq, z_s_eq = self.mooring_equations()
        x_f_sym, z_f_sym = sp.symbols('x_f z_f')
        x_f_eq = x_f_eq.subs(x_f_sym, self.x_f)
        z_f_eq = z_f_eq.subs(z_f_sym, self.z_f)
        
        # Build the vector of functions
        func_vector = sp.Matrix([x_f_eq, z_f_eq])
    
        # Compute the Jacobian matrix with respect to Vf and Hf
        jac_matrix = func_vector.jacobian([self.Vf, self.Hf])
    
        # Convert the functions and Jacobian to lambda functions for usage with scipy
        func_vector_lambda = [sp.lambdify([self.Vf, self.Hf], func) for func in func_vector]
        jac_matrix_lambda = sp.lambdify([self.Vf, self.Hf], jac_matrix)

        # Initial guess
        if initial_guess is None:
            initial_guess = [self.HF0, self.VF0]
        
        # Define the system of mooring proile equations to solve
        def func_to_solve(x):
            return [f(*x) for f in func_vector_lambda]
    
        # Define the Jacobian of the system of equations
        def jac_to_solve(x):
            return jac_matrix_lambda(*x)
    
        # Use scipy's root function with lm method
        solution = optimize.root(func_to_solve, initial_guess, jac=jac_to_solve)
        Vf_sol, Hf_sol = solution.x
        
        # STEP 2: plot mooring line profile ----------------------------------
               
        # Generate s values
        s_values_chain = np.linspace(0, self.l1, self.profile_res)
                        
        # Calculate the corresponding x_s and z_s values
        x_s_chain = [x_s_eq.subs({self.s: s_val, self.Hf: Hf_sol, self.Vf: Vf_sol}).evalf() for s_val in s_values_chain]
        z_s_chain = [z_s_eq.subs({self.s: s_val, self.Hf: Hf_sol, self.Vf: Vf_sol}).evalf() for s_val in s_values_chain]
                  
        lrd_x = LRD_x.subs({self.Hf: Hf_sol, self.Vf: Vf_sol}).evalf()
        lrd_z = LRD_z.subs({self.Hf: Hf_sol, self.Vf: Vf_sol}).evalf()
        
        x_s_lrd = [np.max(x_s_chain), np.max(x_s_chain) + lrd_x]
        z_s_lrd = [np.max(z_s_chain), np.max(z_s_chain) + lrd_z]
        
        # Print & Plot stuff if debugging is on
        
        if self.debugging == True:
                      
            print('Hf final = ' + str(Hf_sol))
            print('Vf final = ' + str(Vf_sol))
            print('Fairlead T  = ' + str(np.sqrt(Hf_sol**2 + Vf_sol**2)))
            
            print('x_f final = ' + str(x_s_lrd[-1]))
            print('z_f final = ' + str(z_s_lrd[-1]))
            
            print('lrd_x final = ' + str(lrd_x))
            print('lrd_z final = ' + str(lrd_z))
            print('lrd stretched length  = ' + str(sp.sqrt(lrd_x**2 + lrd_z**2)))
                    
            # Plot the data
            plt.figure(figsize=(10, 6))
            plt.plot(x_s_chain, z_s_chain, label='Chain')
            plt.plot(x_s_lrd, z_s_lrd, label='LRD', c='r')
            plt.xlabel('x_s')
            plt.ylabel('z_s')
            plt.legend()
            plt.grid(False)
            plt.show()
                
        return solution.x , x_s_chain, z_s_chain, x_s_lrd, z_s_lrd
    
    
    def tension_offset(self, x_f_eq, z_f_eq, x_s_eq, z_s_eq, LRD_x, LRD_z):
        
            # Initialize list to store results
        tension_values = []
        
        # Generate displacement values
        displacement_values = displacement_values = np.arange(0, self.max_offset + self.offset_res, self.offset_res)
        
        # Initial guess
        initial_guess = [self.HF0, self.VF0]
        initial_x_f = self.x_f
    
        for displacement in displacement_values:
            # Update x_f for the new displacement
            self.x_f = initial_x_f + displacement
    
            # Solve system with new x_f and initial_guess
            result, _, _, _, _ = self.solve_system(x_f_eq, z_f_eq, x_s_eq, z_s_eq, LRD_x, LRD_z, initial_guess)
    
            # Compute tension
            Hf_sol, Vf_sol = result
            tension = np.sqrt(Hf_sol**2 + Vf_sol**2)
    
            # Append tension to results list
            tension_values.append(tension)
    
            # Use the solution from this iteration as the initial guess for the next iteration
            initial_guess = [Hf_sol, Vf_sol]
    
        # After loop, plot the results
        plt.figure(figsize=(10, 6))
        plt.plot(displacement_values, tension_values)
        plt.xlabel('Displacement')
        plt.ylabel('Tension')
        plt.grid(True)
        plt.show()
    
        return displacement_values, tension_values

        

# Create an instance of the mooring_line class
ml = mooring_line(x_f=796.73, z_f=136, l1=815.35, ea1=753e3, lrd_type='TFI', 
                  mooring_type='Catenary', w1=1.390, rated_tension = 2000,
                  rated_strain = 0.5, lrd_length = 10, max_offset = 20,
                  debugging=False)

# Initiate the system equations
x_f_eq, z_f_eq, x_s_eq, z_s_eq, LRD_x,LRD_z = ml.mooring_equations()
# Use the solve_system function to find the fairlead tensions (timed)
start_time = timeit.default_timer()
(Vf,Hf), x_s_chain, z_s_chain,  x_s_lrd, z_s_lrd = ml.solve_system(x_f_eq, z_f_eq, x_s_eq, z_s_eq, LRD_x,LRD_z)
end_time = timeit.default_timer()
execution_time = end_time - start_time
print(f"Executed the function in: {execution_time} seconds")

# Use the tension_offset function to find the t-o profile (timed)
start_time = timeit.default_timer()
displacement_values, tension_values = ml.tension_offset(x_f_eq, z_f_eq, x_s_eq, z_s_eq, LRD_x, LRD_z)
end_time = timeit.default_timer()
execution_time = end_time - start_time
print(f"Executed the function in: {execution_time} seconds")

# ============== For performance monitoring ===================================

# with cProfile.Profile() as pr:
#     ml.solve_system(x_f_eq, z_f_eq, x_s_eq, z_s_eq, LRD_x,LRD_z)
# 
# df = pd.DataFrame(
#     pr.getstats(),
#     columns=['func', 'ncalls', 'ccalls', 'tottime', 'cumtime', 'callers'])
# 
# # Print the fairlead tensions
# print('Vertical tension at the fairlead:', Vf, 'kN')
# print('Horizontal tension at the fairlead:', Hf, 'kN')
# =============================================================================
