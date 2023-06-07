# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 11:57:55 2023

@author: ogf1n20
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.interpolate import make_interp_spline, BSpline
from scipy.optimize import curve_fit
import pandas as pd
import numpy as np

class LrdDesign:
    """Class for designing and displaying LRDs."""
    
    def __init__(self, lrd_type, *args, **kwargs):   
        """
        Initialize LrdDesign with LRD type and additional parameters.

        Parameters:
            lrd_type: type of LRD, which can be either TFI, DO
            **kwargs: additional parameters, depending on the lrd_type
        """
        
        # General LRD inputs for both technologies
        self.lrd_type = lrd_type  # type of LRD, which can be either TFI, DO
        # Length of the spring for TFI, or length of the cylinder for DO
        self.lrd_length = kwargs.get('lrd_length') 
        # Diameter of the spring for TFI, length of the cylinder for DO         
        self.lrd_dia = kwargs.get('lrd_dia')
           
        # Additional arguments depending on the lrd_type
        if self.lrd_type == "DO":
            self.lrd_o = kwargs.get('lrd_o')  # Additional parameter for DO lrd_type
            self.lrd_v = kwargs.get('lrd_v')  # Additional parameter for DO lrd_type
            self.lrd_fg = kwargs.get('lrd_fg')  # Additional parameter for DO lrd_type
            self.theta = kwargs.get('theta')  # Additional parameter for DO lrd_type

        elif self.lrd_type == "TFI":
            self.rated_strain = kwargs.get('rated_strain') 
            self.rated_tension = kwargs.get('rated_tension')    
            
    def _arccot(self, x):
        return np.pi / 2 - np.arctan(x)
    
    def _get_dublin_stiffness(self, x):
        """Calculate alpha and extension based on input T."""
        theta_rad = np.radians(self.theta)
        alpha = self._arccot((self.lrd_o * self.lrd_fg - x * self.lrd_v * np.cos(theta_rad)) / (x * self.lrd_v * np.sin(theta_rad)))
        extension = - np.cos(theta_rad + alpha) * self.lrd_v + np.cos(theta_rad) * self.lrd_v
        return alpha, extension
    
    def _get_tfi_stiffness(self, x):
        """tfi stiffness eqn"""
        
        A,B,C,D,J = 0.0123801, 211.947, 0.751564, 11.4939, -16715.3 # fixed constants
        
        rt = 485.82 / self.rated_tension
        rs = self.rated_strain / 3.5885 
        
        term1 = (A * ( rt * x - J) - B) / (1 + (A * (rt * x - J) - B - C) ** 2)
        term2 = (A * J + B) / (1 + (- A * J - B - C) ** 2)
        term3 = D * np.sqrt(A * (rt * x - J))
        term4 = - D * np.sqrt(-1 * A * J)
        
        return rs * (term1 + term2 + term3 + term4)
      
                        
    def draw_plot_lrd(self):
        
        if self.lrd_type == "DO":
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
            
            # Draw the lrd ---------------------------------------------------
            
            # Create the rectangle
            full_rectangle = plt.Rectangle((0, 0), self.lrd_dia, self.lrd_length, linewidth=1.5, edgecolor='black', facecolor='none')
            ax1.add_patch(full_rectangle)
            bottom_rectangle = plt.Rectangle((0, 0), self.lrd_dia, self.lrd_length / 2 - self.lrd_o /2,
                                             linewidth=0.5, edgecolor='black', facecolor='gray', alpha = 0.2)
            ax1.add_patch(bottom_rectangle)
    
            # Calculate CoB (Center of Buoyancy) and CoG (Center of Gravity)
            CoB = [self.lrd_dia / 2, self.lrd_length / 2]
            CoG = [self.lrd_dia / 2, self.lrd_length / 2 - self.lrd_o]
            
            # Calculate hinge points
            top_hinge = [self.lrd_dia / 2, self.lrd_length / 2 + self.lrd_v / 2]
            bottom_hinge = [self.lrd_dia / 2, self.lrd_length / 2 - self.lrd_v / 2]
    
            # Plotting CoB and CoG points
            ax1.scatter(CoB[0], CoB[1], s=3, c='k')
            ax1.text(CoB[0], CoB[1] + 0.015*self.lrd_length, 'CoB', ha='center', fontsize=8)
            ax1.scatter(CoG[0], CoG[1], s=3, c='k')
            ax1.text(CoG[0], CoG[1] + 0.015*self.lrd_length, 'CoG', ha='center', fontsize=8)
            
            # Plotting hinge points
            ax1.scatter(top_hinge[0], top_hinge[1], s=14, facecolor = 'None', 
                       edgecolor='k', linewidth = 0.2)
            ax1.scatter(bottom_hinge[0], bottom_hinge[1], s=14, facecolor = 'None', 
                       edgecolor='k', linewidth = 0.2)
            
            
            # Dimension lines
            ax1.annotate("", xy=(0, -0.15*self.lrd_dia), xycoords='data', xytext=(self.lrd_dia, -0.15*self.lrd_dia), textcoords='data',
                        arrowprops=dict(arrowstyle='<->', lw=0.4, ls='solid'))
            ax1.text(self.lrd_dia / 2, -0.15*self.lrd_length, f'D = {self.lrd_dia} m', ha='center', fontsize = 6)
    
            ax1.annotate("", xy=(-0.15*self.lrd_dia, 0), xycoords='data', xytext=(-0.15*self.lrd_dia, self.lrd_length), textcoords='data',
                        arrowprops=dict(arrowstyle='<->', lw=0.4, ls='solid'))
            ax1.text(-0.2*self.lrd_dia, self.lrd_length / 2, f'L = {self.lrd_length} m', ha='right', va='center', rotation='vertical', fontsize = 6)
                        
            # Annotations for O
            ax1.annotate("", xy=(1.2*self.lrd_dia, CoG[1]), xycoords='data', xytext=(1.2*self.lrd_dia, CoB[1]), textcoords='data',
            arrowprops=dict(arrowstyle='<->', lw=0.4, ls='solid'))
            ax1.text(1.2*self.lrd_dia, CoB[1] + 0.01*self.lrd_length, f'O = {self.lrd_o} m',  ha='center', va='bottom', rotation='vertical', fontsize=6)
            
            # Annotations for V
            ax1.annotate("", xy=(1.35*self.lrd_dia, top_hinge[1]), xycoords='data', xytext=(1.35*self.lrd_dia, bottom_hinge[1]), textcoords='data',
                        arrowprops=dict(arrowstyle='<->', lw=0.4, ls='solid'))
            ax1.text(1.35*self.lrd_dia, top_hinge[1] + 0.01*self.lrd_length, f'V = {self.lrd_v} m', ha='center', va='bottom', rotation='vertical', fontsize=6)
    
            # Arrows at new points
            ax1.arrow(top_hinge[0], top_hinge[1], -self.lrd_dia/3*np.cos(np.radians(self.theta)),
                     -self.lrd_dia/3*np.sin(np.radians(self.theta)), fc='k', ec='k', head_width=0.15, head_length=0.25)
            ax1.arrow(bottom_hinge[0], bottom_hinge[1], self.lrd_dia/3*np.cos(np.radians(self.theta)),
                     self.lrd_dia/3*np.sin(np.radians(self.theta)), fc='k', ec='k', head_width=0.15, head_length=0.25)
    
            # Set the aspect of the plot to be equal and hide ax1es
            ax1.set_aspect('equal')
            ax1.axis('off')
            
            # Expand plot limits
            ax1.set_xlim(-0.25*self.lrd_dia, 1.5*self.lrd_dia)
            ax1.set_ylim(-0.25*self.lrd_length, 1.25*self.lrd_length)
    
            # Plot the lrd cuve ----------------------------------------------
            
            # Generate 100 evenly spaced T values between T_min and T_max1
            T_values = np.linspace(0, self.lrd_fg, 100)            
            # Calculate ΔL for each T value
            extension_values = [self._get_dublin_stiffness(T)[1] for T in T_values]
                
            # Create the plot deltaL
            ax2.plot(extension_values, T_values)
            ax2.set_xlabel('LRD extension (m)')
            ax2.set_ylabel('Tension (N)')
            ax2.set_title('LRD stiffness curve')
    
            plt.tight_layout()
                        
            return fig
        
        if self.lrd_type == "TFI":
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
            img = mpimg.imread(r'C:\Users\ogf1n20\Documents\03-PYTHON\VISIMOOR\tfi_seaspring.png')
            ax1.imshow(img)
            
            # Read the CSV file & sort by strain values
            df = pd.read_csv(r'C:\Users\ogf1n20\Documents\03-PYTHON\VISIMOOR\tfi_original_curve.csv',
                             header=None)
            
            # Get the data for the x and y axes
            strain = df.iloc[:, 0]  # Assuming the first column is the x-axis
            tension = df.iloc[:, 1]  # Assuming the second column is the y-axis
            
            # Scale tension data with rated tension and strain
            tension = tension * (self.rated_tension / 1000)
            strain = strain * (self.rated_strain / 0.5)
            
            # Make the line smooth
            strain_smooth = np.linspace(strain.min(), self.rated_strain + 0.05, 500) 
            spl = make_interp_spline(strain, tension, k=3)  # type: BSpline
            tension_smooth = spl(strain_smooth)
            
            # Plot the data
            ax2.plot(strain_smooth, tension_smooth)
                 
            strain_fitted = [self._get_tfi_stiffness(x) for x in tension_smooth]
                 
            ax2.plot(strain_fitted, tension_smooth, 'g--', label='Initial guess')
            ax2.set_xlim(0,self.rated_strain * 1.1)
            ax2.set_ylim(0,self.rated_tension * 1.25)
            
            # Add a legend to the plot
            ax2.legend()            
            # Show the plot
            plt.show()
            
#============================================================================
dublin_example = LrdDesign("DO", lrd_dia=4, lrd_length=12, lrd_o=1.5, lrd_v=5, theta = 40, lrd_fg = 5e6)
dublin_example.draw_plot_lrd()
#============================================================================
tfi_example = LrdDesign("TFI", lrd_dia=4, lrd_length=12, rated_strain=0.5, rated_tension= 3000)
tfi_example.draw_plot_lrd()