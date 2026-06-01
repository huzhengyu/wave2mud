# -*- coding: utf-8 -*-
"""
Created on Wed Apr 23 13:48:28 2025

@author: HZY
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.interpolate import griddata
from matplotlib.colors import LogNorm
from matplotlib import cm
import seaborn as sns
from scipy.interpolate import interp1d
from matplotlib.ticker import FuncFormatter

# Define formatter function
def format_with_unicode_minus(x, pos):
    return f'{x:.1f}'.replace('-', '\u2212')  # \u2212 is the true minus sign

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
d = 0.06 #mud thickness 
rho_b = 1420 #mud bulk density
nu_0 = 0.45
nu = 0.00211 #best-fit viscosity

wavelength = 1.10846 #water wave length
x_0 = 5.42 #WG 1 location
a = 0.02
T = 0.9
omega = 2*np.pi/T
delta = np.sqrt(2*nu/omega) #boundary layer thickness
rhow = 1000
g = 9.81
E0 = 1/2*rhow*g*a**2
h = 0.24
k = 2*np.pi/wavelength
cg = 1/2* wavelength/T * (1+2*k*h/np.sinh(2*k*h)) 

Num = pd.read_excel('../data/SourceData.xlsx',sheet_name = 'Figure4')

X_mean = np.array([5.24,13.35])
Z_mean = np.array([0,0])+0.06


x = Num["mud_field_x"].dropna()
z = Num["mud_field_z"].dropna()+0.06
xi = np.linspace(min(x), max(x), 1000)
zi = np.linspace(min(z), max(z), 200)
X, Z = np.meshgrid(xi, zi)

x_newtonian = Num["Newtonian_mud_field_x"].dropna()
z_newtonian = Num["Newtonian_mud_field_z"].dropna()+0.06
xi_newtonian = np.linspace(min(x_newtonian), max(x_newtonian), 1000)
zi_newtonian = np.linspace(min(z_newtonian), max(z_newtonian), 200)
X_newtonian, Z_newtonian = np.meshgrid(xi_newtonian, zi_newtonian)

convection = Num["convectiveTermMean"].dropna()
pressureT = Num["pressureTransportMean"].dropna()
viscousDif = Num["viscousDiffusionMean"].dropna()
viscousDis = Num["viscousDissipationMean"].dropna()

convection_newtonian = Num["Newtonian_convectiveTermMean"].dropna()
pressureT_newtonian = Num["Newtonian_pressureTransportMean"].dropna()
viscousDif_newtonian = Num["Newtonian_viscousDiffusionMean"].dropna()
viscousDis_newtonian = Num["Newtonian_viscousDissipationMean"].dropna()

convection_interp = griddata((x, z), convection, (X, Z), method='linear')
pressureT_interp = griddata((x, z), pressureT, (X, Z), method='linear')
viscousDif_interp = griddata((x, z), viscousDif, (X, Z), method='linear')
viscousDis_interp = griddata((x, z), viscousDis, (X, Z), method='linear')

convection_newtonian_interp = griddata((x_newtonian, z_newtonian), convection_newtonian, (X_newtonian, Z_newtonian), method='linear')
pressureT_newtonian_interp = griddata((x_newtonian, z_newtonian), pressureT_newtonian, (X_newtonian, Z_newtonian), method='linear')
viscousDif_newtonian_interp = griddata((x_newtonian, z_newtonian), viscousDif_newtonian, (X_newtonian, Z_newtonian), method='linear')
viscousDis_newtonian_interp = griddata((x_newtonian, z_newtonian), viscousDis_newtonian, (X_newtonian, Z_newtonian), method='linear')

x_Case2B = Num["x_Case2B"].dropna()
epsilonD_Case2B = Num["epsilonD_Case2B"].dropna()
x_Case2C = Num["x_Case2C"].dropna()
epsilonD_Case2C = Num["epsilonD_Case2C"].dropna()
x_Case4C = Num["x_Case4C"].dropna()
epsilonD_Case4C = Num["epsilonD_Case4C"].dropna()

mean_interp = interp1d(X_mean, Z_mean, bounds_error=False, fill_value=np.nan)
Z_mean_on_grid = mean_interp(X[0, :])


Z_mean_grid = np.tile(Z_mean_on_grid, (Z.shape[0], 1))  # repeat to shape of P_interp
mask = Z > Z_mean_grid  # mask values above surface

convection_interp_masked = np.ma.masked_where(mask, convection_interp)
pressureT_interp_masked = np.ma.masked_where(mask, pressureT_interp)
viscousDif_interp_masked = np.ma.masked_where(mask, viscousDif_interp)
viscousDis_interp_masked = np.ma.masked_where(mask, viscousDis_interp)

convection_newtonian_interp_masked = np.ma.masked_where(mask, convection_newtonian_interp)
pressureT_newtonian_interp_masked = np.ma.masked_where(mask, pressureT_newtonian_interp)
viscousDif_newtonian_interp_masked = np.ma.masked_where(mask, viscousDif_newtonian_interp)
viscousDis_newtonian_interp_masked = np.ma.masked_where(mask, viscousDis_newtonian_interp)

# Create a figure with subplots
fig = plt.figure(figsize=(12, 10))
gs = gridspec.GridSpec(4, 3, figure=fig, wspace=0.3, hspace=0.9)

"""
Convective term
"""
ax1 = fig.add_subplot(gs[0, 0])
contour1 = ax1.contourf((X-x_0)/wavelength, Z/d, convection_interp_masked/(rhow*a**2*omega**3), levels=np.linspace(-0.0002, 0.001, 31), cmap = 'plasma')  # Line contours

# Get position of the main axis
pos = ax1.get_position()
dy = -0.025  # adjust as needed
ax1.set_position([pos.x0, pos.y0 + dy, pos.width, pos.height])

pos = ax1.get_position()
# Define a new axis for the colorbar just above the main plot
cbar_ax = fig.add_axes([pos.x0, pos.y1 + 0.02, pos.width, 0.02])  # [left, bottom, width, height]

cbar = plt.colorbar(contour1, cax=cbar_ax, orientation='horizontal')

ticks = np.linspace(0, 0.001, 3)  # 5 evenly spaced ticks from min to max
cbar.set_ticks(ticks)

cbar.set_label('$\overline{C_k}$', labelpad=5)  # Optional: label for the color bar
cbar.ax.xaxis.set_label_position('top')
cbar.ax.tick_params(labelsize=label_font_size)  # <<-- Set tick label font size here

ax1.plot((X_mean-x_0)/wavelength, Z_mean/d, color='#866E65')
ax1.set_xlim(0,7)
ax1.set_ylim(0, 1.15)
ax1.set_xticks([0, 2, 4, 6])
ax1.set_ylabel("$z/d$")

for c in contour1.collections:
    c.set_rasterized(True)
    
"""
Transport of pressure
"""
ax2 = fig.add_subplot(gs[0, 1])
contour2 = ax2.contourf((X-x_0)/wavelength, Z/d, pressureT_interp_masked/(rhow*a**2*omega**3), levels=np.linspace(0, 0.03, 31), cmap = 'plasma')  # Line contours

# Get position of the main axis
pos = ax2.get_position()
dy = -0.025  # adjust as needed
ax2.set_position([pos.x0, pos.y0 + dy, pos.width, pos.height])

pos = ax2.get_position()

# Define a new axis for the colorbar just above the main plot
cbar_ax = fig.add_axes([pos.x0, pos.y1 + 0.02, pos.width, 0.02])  # [left, bottom, width, height]

cbar = plt.colorbar(contour2, cax=cbar_ax, orientation='horizontal')

ticks = np.linspace(0, 0.03, 4)  # 5 evenly spaced ticks from min to max
cbar.set_ticks(ticks)

cbar.set_label(r'$\overline{T_p}$', labelpad=5)  # Optional: label for the color bar
cbar.ax.xaxis.set_label_position('top')
cbar.ax.tick_params(labelsize=label_font_size)  # <<-- Set tick label font size here

ax2.plot((X_mean-x_0)/wavelength, Z_mean/d, color='#866E65')
ax2.set_xlim(0,7)
ax2.set_ylim(0, 1.15)
ax2.set_xticks([0, 2, 4, 6])
ax2.set_ylabel("$z/d$")

for c in contour2.collections:
    c.set_rasterized(True)

"""
Time-averaged Viscous diffusion
"""
ax3 = fig.add_subplot(gs[1, 0])
contour3 = ax3.contourf((X-x_0)/wavelength, Z/d, viscousDif_interp_masked/(rhow*a**2*omega**3), levels=np.linspace(-0.04, 0.08, 31), cmap = 'plasma')  # Line contours

# Get position of the main axis
pos = ax3.get_position()

# Define a new axis for the colorbar just above the main plot
cbar_ax = fig.add_axes([pos.x0, pos.y1 + 0.02, pos.width, 0.02])  # [left, bottom, width, height]

cbar = plt.colorbar(contour3, cax=cbar_ax, orientation='horizontal')
cbar.set_label(r'$\overline{D_v}$', labelpad=5)  # Optional: label for the color bar
cbar.ax.xaxis.set_label_position('top')
cbar.ax.tick_params(labelsize=label_font_size)  # <<-- Set tick label font size here

ticks = np.linspace(-0.04, 0.08, 4)  # 5 evenly spaced ticks from min to max
cbar.set_ticks(ticks)

ax3.plot((X_mean-x_0)/wavelength, Z_mean/d, color='#866E65')
ax3.set_xlim(0,7)
ax3.set_ylim(0, 1.15)
ax3.set_xticks([0, 2, 4, 6])
ax3.set_xlabel("$x/\lambda$")
ax3.set_ylabel("$z/d$")

for c in contour3.collections:
    c.set_rasterized(True)
    
"""
Time-averaged Viscous dissipation rate
"""
ax4 = fig.add_subplot(gs[1, 1])
contour4 = ax4.contourf((X-x_0)/wavelength, Z/d, viscousDis_interp_masked/(rhow*a**2*omega**3), levels=np.linspace(0, 0.1, 31), linewidths=0, antialiased=False, cmap = 'plasma')  # Line contours

# Get position of the main axis
pos = ax4.get_position()

# Define a new axis for the colorbar just above the main plot
cbar_ax = fig.add_axes([pos.x0, pos.y1 + 0.02, pos.width, 0.02])  # [left, bottom, width, height]

cbar = plt.colorbar(contour4, cax=cbar_ax, orientation='horizontal')
cbar.set_label(r'$\overline{\epsilon}$')  # Optional: label for the color bar
cbar.ax.xaxis.set_label_position('top')
cbar.ax.tick_params(labelsize=label_font_size)  # <<-- Set tick label font size here

ax4.plot((X_mean-x_0)/wavelength, Z_mean/d, color='#866E65')
ax4.set_xlim(0,7)
ax4.set_ylim(0, 1.15)
ax4.set_xticks([0, 2, 4, 6])
ax4.set_xlabel("$x/\lambda$")
ax4.set_ylabel("$z/d$")

ticks = np.linspace(0, 0.1, 6)  # 5 evenly spaced ticks from min to max
cbar.set_ticks(ticks)

for c in contour4.collections:
    c.set_rasterized(True)

"""
Depth-integrated Time-averaged Viscous dissipation
"""
ax5 = fig.add_subplot(gs[0:2, 2])

# Get position of the main axis
pos = ax5.get_position()
dy = -0.025  # adjust as needed
ax5.set_position([pos.x0, pos.y0, pos.width, pos.height+dy])

eps = viscousDis_interp_masked
eps_newtonian = viscousDis_newtonian_interp_masked

# # Fill masked values with 0 (important for integration)
eps_filled = np.ma.filled(eps, 0.0)
eps_newtonian_filled = np.ma.filled(eps_newtonian, 0.0)

dx = xi[1] - xi[0]
dz = zi[1] - zi[0]
dx_newtonian = xi_newtonian[1] - xi_newtonian[0]
dz_newtonian = zi_newtonian[1] - zi_newtonian[0]

D_depth = np.sum(eps_filled, axis=1) * dx
D_cum = np.cumsum(D_depth) * dz
D_cum_norm = D_cum / D_cum[-1]

D_depth_newtonian = np.sum(eps_newtonian_filled, axis=1) * dx_newtonian
D_cum_newtonian = np.cumsum(D_depth_newtonian) * dz_newtonian
D_cum_norm_newtonian = D_cum_newtonian / D_cum_newtonian[-1]

ax5.plot(D_cum_norm, zi/d, '-', color='#1f77b4', markersize=3, label='Non-Newtonian')
ax5.plot(D_cum_norm_newtonian, zi_newtonian/d, '--', color='#DB494E', markersize=3, label='Newtonian')

ax5.plot([-1,2],[delta/d,delta/d], linestyle=':', color='gray')
ax5.text(0.1, 0.475, '$z=\delta$', 
         transform=ax5.transAxes, verticalalignment='top')
ax5.set_xlim([-0.05, 1.1])
ax5.set_ylim([-0.05, 1.1])
sns.despine(ax=ax5)
ax5.set_xlabel("Cumulative dissipation fraction")
ax5.set_ylabel("$z/d$")
ax5.legend(frameon=False, fontsize=label_font_size)


ax6 = fig.add_subplot(gs[2, 0])

# Get position of the main axis
pos = ax6.get_position()
dy = 0.03  # adjust as needed
ax6.set_position([pos.x0, pos.y0, pos.width, pos.height+dy])

mask_outerLayer = Z > delta
mask_boundaryLayer = ~mask_outerLayer

mask_newtonian_outerLayer = Z_newtonian > delta
mask_newtonian_boundaryLayer = ~mask_outerLayer

D_boundaryLayer_x = np.sum(eps_filled * mask_boundaryLayer, axis=0) * dz
D_newtonian_boundaryLayer_x = np.sum(eps_newtonian_filled * mask_newtonian_boundaryLayer, axis=0) * dz
D_outerLayer_x    = np.sum(eps_filled * mask_outerLayer, axis=0) * dz
D_newtonian_outerLayer_x    = np.sum(eps_newtonian_filled * mask_newtonian_outerLayer, axis=0) * dz

ax6.plot((xi-x_0)/wavelength, D_boundaryLayer_x / (E0*cg),'-', color='#1f77b4', label='Non-Newtonian')
ax6.plot((xi_newtonian-x_0)/wavelength, D_newtonian_boundaryLayer_x / (E0*cg),'--', color='#DB494E', label='Newtonian')

ax6.set_xlim(0,7)
ax6.set_ylim(0.03,0.08)
sns.despine(ax=ax6)
ax6.set_xlabel("$x/\lambda$")
ax6.set_ylabel(r"$\epsilon_{D,z \leq \delta}$")
ax6.legend(frameon=False, fontsize=label_font_size)

ax7 = fig.add_subplot(gs[2, 1])
# Get position of the main axis
pos = ax7.get_position()
dy = 0.03  # adjust as needed
ax7.set_position([pos.x0, pos.y0, pos.width, pos.height+dy])

ax7.plot((xi-x_0)/wavelength, D_outerLayer_x / (E0*cg),'-', color='#1f77b4', label='Non-Newtonian')
ax7.plot((xi_newtonian-x_0)/wavelength, D_newtonian_outerLayer_x / (E0*cg),'--', color='#DB494E', label='Newtonian')

ax7.set_xlim(0,7)
ax7.set_yticks([0.01,0.02])
sns.despine(ax=ax7)
ax7.set_xlabel("$x/\lambda$")
ax7.set_ylabel(r"$\epsilon_{D,z>\delta}$")
ax7.legend(frameon=False, fontsize=label_font_size)

ax8 = fig.add_subplot(gs[2, 2])

# Get position of the main axis
pos = ax8.get_position()
dy = 0.03  # adjust as needed
ax8.set_position([pos.x0, pos.y0, pos.width, pos.height+dy])

ax8.plot(x_Case2B, epsilonD_Case2B, color='black', label = 'Case 2B')
ax8.plot(x_Case2C, epsilonD_Case2C, color='#DB494E', linestyle = '--', label = 'Case 2C')
ax8.plot(x_Case4C, epsilonD_Case4C, color='#1f77b4', linestyle = '-.', label = 'Case 4C')
ax8.legend(frameon=False, fontsize=label_font_size)

ax8.set_xlim(0,7)
ax8.set_ylim(0.04, 0.1)
ax8.set_xticks([0, 2, 4, 6])
ax8.set_yticks(np.linspace(0.04, 0.1, 4))
sns.despine(ax=ax8)
ax8.set_xlabel("$x/\lambda$")
ax8.set_ylabel("$\epsilon_D$")

axes = [ax1, ax5, ax6, ax7, ax8]
labels = list("abcde")
    
for ax, label in zip(axes, labels):
    # Convert the top-left of each axis to figure coordinates
    inv = fig.transFigure.inverted()
    x_fig, y_fig = inv.transform(ax.transAxes.transform((0, 1)))
    
    # Apply a small offset
    fig.text(x_fig - 0.0425, y_fig+0.01, f"{label}", fontsize=15, fontweight='bold')
 
plt.savefig("Figure4.pdf", dpi=1200, bbox_inches='tight', pad_inches=0, facecolor='white')
plt.show()
