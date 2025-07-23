# -*- coding: utf-8 -*-
"""
Created on Fri Apr 18 13:35:04 2025

@author: HZY
"""
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.interpolate import griddata
from matplotlib.colors import LogNorm
from matplotlib import cm

plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 12,
    'axes.linewidth': 1,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.major.size': 4,
    'ytick.major.size': 4,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'lines.linewidth': 1.5,
    'lines.markersize': 4,
    'legend.frameon': False,
})
label_font_size = 10

Num = pd.read_excel('../data/SourceData.xlsx',sheet_name = 'Figure2')
Exp = pd.read_excel('../data/SourceData.xlsx',sheet_name = 'Hsu2013') 

gamma = Exp["gamma_Hsu13"].dropna()
mu_28 = Exp["mu_Hsu13_phi.28"].dropna()
mu_25 = Exp["mu_Hsu13_phi.25"].dropna()
mu_19 = Exp["mu_Hsu13_phi.19"].dropna()
mu_13 = Exp["mu_Hsu13_phi.13"].dropna()
x = Num["x"].dropna()
y = Num["z"].dropna()
strainRate = Num["Strain Rate"].dropna()
nu = Num["Dynamic Viscosity"].dropna()

xi = np.linspace(min(x), max(x), 1000)
yi = np.linspace(min(y), max(y), 200)
X, Y = np.meshgrid(xi, yi)

# Interpolate z values onto the grid
strainRate = griddata((x, y), strainRate, (X, Y), method='cubic')
mu_model = griddata((x, y), nu, (X, Y), method='cubic')*1420

# Create a figure with subplots
fig = plt.figure(figsize=(12, 7))
gs = gridspec.GridSpec(6, 2, figure=fig, wspace=0.1, hspace=0.5)

# Plot the schematic in the first subplot
ax1 = fig.add_subplot(gs[:6, 0:2])

schematic = Image.open('wave_tank.png')
ax1.imshow(schematic)
ax1.axis('off')
ax1.text(-0.05, 0.95, "a", fontsize=12, fontweight='bold', transform=ax1.transAxes)
ax1.text(0.01, 0.35, 'Wave maker', color='k', rotation=-4, fontsize=label_font_size, transform=ax1.transAxes)  
ax1.text(0.95, 0.6, 'Beach', color='k', rotation=-4, fontsize=label_font_size, transform=ax1.transAxes)  
ax1.text(0.45, 0.125, 'Mud', color='k', rotation=-4, fontsize=label_font_size, transform=ax1.transAxes)  

ax1.annotate('Wave gauges', xy=(2060, 210), xytext=(1950, 50),
            arrowprops=dict(arrowstyle='->', color='grey'),
            fontsize=label_font_size)
ax1.annotate(' ', xy=(2350, 230), xytext=(2200, 70),
            arrowprops=dict(arrowstyle='->', color='grey'),
            fontsize=label_font_size)

ax1.annotate('Velocimetry', xy=(2780, 380), xytext=(2780, 200),
            arrowprops=dict(arrowstyle='->', color='grey'),
            fontsize=label_font_size)

ax1.annotate('', xy=(1600, 902), xytext=(20, 750),
            arrowprops=dict(arrowstyle='->', color='grey'),
            fontsize=label_font_size)
ax1.text(0.1, 0.15, 'Distance $x$', color='grey', rotation=-4, fontsize=label_font_size, transform=ax1.transAxes)  

# Plot something else in the second subplot
ax2 = fig.add_subplot(gs[4:6, 0])
ax2.plot(gamma, mu_28, color = '#1f77b4', label = '$\phi=0.28$')
ax2.plot(gamma, mu_25, color = '#DB494E',  label = '$\phi=0.25$')
ax2.plot(gamma, mu_19, color = 'k',  label = '$\phi=0.19$')
ax2.plot(gamma, mu_13, color = 'gray',  label = '$\phi=0.13$')
ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_xlim(2e-3,1e2)
ax2.set_xlabel("$\dot{\gamma}$ (s$^{-1}$)", size= label_font_size)
ax2.set_ylabel("$\mu$ (Pa$\cdot$s)", size= label_font_size)
ax2.text(-0.11, 0.95, "b", fontsize=12, fontweight='bold',transform=ax2.transAxes)
ax2.legend(fontsize=11, ncol=2)
ax2.tick_params(axis='both', which='both',
           direction='in',
           bottom=True, left=True, 
           width=0.8, length=4, color='black',labelsize=label_font_size)

ax3 = fig.add_subplot(gs[4, 1])
ax3.contour = plt.contourf(X, Y, strainRate, levels=30, cmap=cm.PuBu_r)  # Line contours
ax3.set_xlim(5.1, 13.6)
ax3.set_ylabel("$z$", size= label_font_size)
ax3.set_xticks([])
ax3.set_yticks([])
ax3.text(-0.06, 0.95, "c", fontsize=12, fontweight='bold',transform=ax3.transAxes)
ax3.set_title("Strain rate",size= label_font_size)

ax4 = fig.add_subplot(gs[5, 1])
ax4.contour = plt.contourf(X, Y, mu_model, levels=30, norm=LogNorm(vmin=mu_model[mu_model>0].min(), vmax=mu_model[mu_model>0].max()),  cmap='viridis')  # Line contours
ax4.set_xlim(5.1, 13.6)
ax4.set_xlabel("$x$", size= label_font_size)
ax4.set_ylabel("$z$", size= label_font_size)
ax4.set_xticks([])
ax4.set_yticks([])
ax4.set_title("Dynamic viscosity",size= label_font_size)

plt.tight_layout()
# plt.savefig("Figure2.pdf", dpi=1200, bbox_inches='tight', pad_inches=0)
plt.show()
