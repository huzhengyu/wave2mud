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
wavelength = 1.10846 #water wave length
x_0 = 5.42 #WG 1 location
a = 0.02
T = 0.9
omega = 2*np.pi/T
rhow = 1000
g = 9.81
E0 = 1/2*rhow*g*a**2
h = 0.24
k = 2*np.pi/wavelength
cg = 1/2* wavelength/T * (1+2*k*h/np.sinh(2*k*h)) 

Num = pd.read_excel('../data/SourceData.xlsx',sheet_name = 'Figure4')

Xw_surface = Num["water_surface_x"].dropna()
Zw_surface = Num["water_surface_z"].dropna()+0.06
X_surface = Num["mud_surface_x"].dropna()
Z_surface = Num["mud_surface_z"].dropna()+0.06
X_mean = np.array([5.24,13.35])
Z_mean = np.array([0,0])+0.06

x_w = Num["entire_field_x"].dropna()
z_w = Num["entire_field_z"].dropna()+0.06
xi_w = np.linspace(min(x_w), max(x_w), 1000)
zi_w = np.linspace(min(z_w), max(z_w), 200)
X_w, Z_w = np.meshgrid(xi_w, zi_w)
strainRate = Num["entire_field_strainRate_Magnitude"].dropna()

x = Num["mud_field_x"].dropna()
z = Num["mud_field_z"].dropna()+0.06
xi = np.linspace(min(x), max(x), 1000)
zi = np.linspace(min(z), max(z), 200)
X, Z = np.meshgrid(xi, zi)

mu = Num["nu.mud"].dropna()*rho_b
muM = Num["nu.mudMean"].dropna()*rho_b
epsilonxx = Num["epsilon_xx"].dropna()
epsilonxz = Num["epsilon_xz"].dropna()
epsilon = Num["epsilon"].dropna()

convection = Num["convectiveTermMean"].dropna()
pressureT = Num["pressureTransportMean"].dropna()
viscousDif = Num["viscousDiffusionMean"].dropna()
viscousDis = Num["viscousDissipationMean"].dropna()

x_Case2B = Num["x_Case2B"].dropna()
epsilonD_Case2B = Num["epsilonD_Case2B"].dropna()
x_Case2C = Num["x_Case2C"].dropna()
epsilonD_Case2C = Num["epsilonD_Case2C"].dropna()
x_Case4C = Num["x_Case4C"].dropna()
epsilonD_Case4C = Num["epsilonD_Case4C"].dropna()
# 
# # Interpolate scattered data to a grid
strainRate_interp = griddata((x_w, z_w), strainRate, (X_w, Z_w), method='linear')
mu_interp = griddata((x, z), mu, (X, Z), method='linear')
muM_interp = griddata((x, z), muM, (X, Z), method='linear')
epsilonxx_interp = griddata((x, z), epsilonxx, (X, Z), method='linear')
epsilonxz_interp = griddata((x, z), epsilonxz, (X, Z), method='linear')
epsilon_interp = griddata((x, z), epsilon, (X, Z), method='linear')

convection_interp = griddata((x, z), convection, (X, Z), method='linear')
pressureT_interp = griddata((x, z), pressureT, (X, Z), method='linear')
viscousDif_interp = griddata((x, z), viscousDif, (X, Z), method='linear')
viscousDis_interp = griddata((x, z), viscousDis, (X, Z), method='linear')

# Step 2: Create a 2D mask where Z > surface elevation at each X
surface_interp = interp1d(X_surface, Z_surface, bounds_error=False, fill_value=np.nan)
Z_surf_on_grid = surface_interp(X[0, :])  # 1D slice of surface for each x in the grid

mean_interp = interp1d(X_mean, Z_mean, bounds_error=False, fill_value=np.nan)
Z_mean_on_grid = mean_interp(X[0, :])

water_interp = interp1d(Xw_surface, Zw_surface, bounds_error=False, fill_value=np.nan)
Z_water_on_grid = water_interp(X_w[0, :])

# Step 2: Create a 2D mask where Z > surface elevation at each X
Z_surface_grid = np.tile(Z_surf_on_grid, (Z.shape[0], 1))  # repeat to shape of P_interp
mask = Z > Z_surface_grid  # mask values above surface

Z_mean_grid = np.tile(Z_mean_on_grid, (Z.shape[0], 1))  # repeat to shape of P_interp
mask2 = Z > Z_mean_grid  # mask values above surface

Z_water_grid = np.tile(Z_water_on_grid, (Z_w.shape[0], 1))  # repeat to shape of P_interp
mask3 = Z_w > Z_water_grid  # mask values above surface

# Step 3: Apply the mask
strainRate_interp_masked = np.ma.masked_where(mask3, strainRate_interp)
mu_interp_masked = np.ma.masked_where(mask, mu_interp)
muM_interp_masked = np.ma.masked_where(mask2, muM_interp)
epsilonxx_interp_masked = np.ma.masked_where(mask, epsilonxx_interp)
epsilonxz_interp_masked = np.ma.masked_where(mask, epsilonxz_interp)
epsilon_interp_masked = np.ma.masked_where(mask, epsilon_interp)

convection_interp_masked = np.ma.masked_where(mask2, convection_interp)
pressureT_interp_masked = np.ma.masked_where(mask2, pressureT_interp)
viscousDif_interp_masked = np.ma.masked_where(mask2, viscousDif_interp)
viscousDis_interp_masked = np.ma.masked_where(mask2, viscousDis_interp)

# Create a figure with subplots
fig = plt.figure(figsize=(12, 10))
gs = gridspec.GridSpec(4, 3, figure=fig, wspace=0.3, hspace=0.9)

# Plot the schematic in the first subplot
ax1 = fig.add_subplot(gs[0, 0])
contour1 = ax1.contourf((X_w-x_0)/wavelength, Z_w/d, strainRate_interp_masked/(a*omega/d), levels=np.linspace(0, 2.5, 31), cmap = cm.PuBu_r)  # Line contours

# Get position of the main axis
pos = ax1.get_position()

# Define a new axis for the colorbar just above the main plot
cbar_ax = fig.add_axes([pos.x0, pos.y1 + 0.027, pos.width, 0.02])  # [left, bottom, width, height]
cbar = plt.colorbar(contour1, cax=cbar_ax, orientation='horizontal')
cbar.set_label('$|\dot{\gamma}|$', labelpad=5)  # Optional: label for the color bar
cbar.ax.xaxis.set_label_position('top')
cbar.ax.tick_params(labelsize=label_font_size)  # <<-- Set tick label font size here

ticks = np.linspace(0, 2.5, 6)  # 5 evenly spaced ticks from min to max
cbar.set_ticks(ticks)

ax1.plot((Xw_surface-x_0)/wavelength, Zw_surface/d, '--', color='#003366')
ax1.plot((X_surface-x_0)/wavelength, Z_surface/d, '-', color='#866E65')
ax1.set_xlim(0,7)
ax1.set_ylim(0, 6)

# Apply to y-axis
ax1.set_yticks([0, 1.0, 5.0])
ax1.yaxis.set_major_formatter(FuncFormatter(format_with_unicode_minus))
ax1.set_xticks([0, 2, 4, 6])
ax1.set_xlabel("$x/\lambda$")
ax1.set_ylabel("$z/d$")

for c in contour1.collections:
    c.set_rasterized(True)

"""
Viscosity
"""

ax2 = fig.add_subplot(gs[0, 1])
vmin = 1e-3
vmax = 1

log_levels = np.logspace(np.log10(vmin), np.log10(vmax), 30)

contour2 = ax2.contourf(
    (X - x_0)/wavelength,
    Z / d,
    mu_interp_masked / (nu_0 * rho_b),
    levels=log_levels,
    norm=LogNorm(vmin=vmin, vmax=vmax),
    cmap='viridis'
)
# contour2 = ax2.contourf((X-x_0)/wavelength, Z/d, mu_interp_masked/(nu_0*rho_b), levels=30, norm=LogNorm(vmin=mu_interp_masked[mu_interp_masked>0].min()/(nu_0*rho_b), vmax=1), cmap = 'viridis')  # Line contours

# Get position of the main axis
pos = ax2.get_position()

# Define a new axis for the colorbar just above the main plot
cbar_ax = fig.add_axes([pos.x0, pos.y1 + 0.027, pos.width, 0.02])  # [left, bottom, width, height]
cbar = plt.colorbar(contour2, cax=cbar_ax, orientation='horizontal')
cbar.set_ticks([1e-3, 1e-2, 1e-1, 1])  # adjust based on your vmin/vmax
cbar.set_label(r'$\mu$', labelpad=5)  # Optional: label for the color bar
cbar.ax.xaxis.set_label_position('top')
cbar.ax.tick_params(labelsize=label_font_size)  # <<-- Set tick label font size here

ax2.plot((X_surface-x_0)/wavelength, Z_surface/d, color='#866E65')
ax2.set_xlim(0,7)
ax2.set_ylim(0, 1.15)
ax2.set_xticks([0, 2, 4, 6])
ax2.set_xlabel("$x/\lambda$")
ax2.set_ylabel("$z/d$")
# ax2.set_yticks(np.linspace(-0.06, 0, 3))

for c in contour2.collections:
    c.set_rasterized(True)

"""
Viscosity Mean
"""
ax3 = fig.add_subplot(gs[0, 2])
contour3 = ax3.contourf((X-x_0)/wavelength, Z/d, muM_interp_masked/(nu_0*rho_b), levels=np.linspace(0, 0.04, 31), cmap = 'viridis')  # Line contours

# Get position of the main axis
pos = ax3.get_position()

# Define a new axis for the colorbar just above the main plot
cbar_ax = fig.add_axes([pos.x0, pos.y1 + 0.027, pos.width, 0.02])  # [left, bottom, width, height]
cbar = plt.colorbar(contour3, cax=cbar_ax, orientation='horizontal')
ticks = np.linspace(0, 0.04, 5)  # 5 evenly spaced ticks from min to max
cbar.set_ticks(ticks)
cbar.set_label(r'$\overline{\mu}$', labelpad=5)  # Optional: label for the color bar
cbar.ax.xaxis.set_label_position('top')
cbar.ax.tick_params(labelsize=label_font_size)  # <<-- Set tick label font size here

ax3.plot((X_mean-x_0)/wavelength, Z_mean/d, color='#866E65')
ax3.set_xlim(0,7)
ax3.set_ylim(0, 1.15)
ax3.set_xticks([0, 2, 4, 6])
ax3.set_xlabel("$x/\lambda$")
ax3.set_ylabel("$z/d$")

for c in contour3.collections:
    c.set_rasterized(True)
    
"""
Viscous dissipation XX
"""
ax4 = fig.add_subplot(gs[1, 0])
contour4 = ax4.contourf((X-x_0)/wavelength, Z/d, epsilonxx_interp_masked/(rhow*a**2*omega**3), levels=30, cmap = 'cividis')  # Line contours
ax4.plot((X_surface-x_0)/wavelength, Z_surface/d, color='#866E65')
ax4.set_xlim(0,7)
ax4.set_ylim(0, 1.15)
ax4.set_xticks([0, 2, 4, 6])
ax4.set_xlabel("$x/\lambda$")
ax4.set_ylabel("$z/d$")

# Get position of the main axis
pos = ax4.get_position()
dy = -0.01  # adjust as needed
ax4.set_position([pos.x0, pos.y0 + dy, pos.width, pos.height])

pos = ax4.get_position()

# Define a new axis for the colorbar just above the main plot
cbar_ax = fig.add_axes([pos.x0, pos.y1 + 0.02, pos.width, 0.02])  # [left, bottom, width, height]
cbar = plt.colorbar(contour4, cax=cbar_ax, orientation='horizontal')
ticks = np.linspace(0, 0.004, 5)  # 5 evenly spaced ticks from min to max
cbar.set_ticks(ticks)
cbar.set_label(r'$\epsilon_{xx}$', labelpad=5)  # Optional: label for the color bar
cbar.ax.xaxis.set_label_position('top')
cbar.ax.tick_params(labelsize=label_font_size)  # <<-- Set tick label font size here

for c in contour4.collections:
    c.set_rasterized(True)

"""
Viscous dissipation XZ
"""
ax5 = fig.add_subplot(gs[1, 1])
contour5 = ax5.contourf((X-x_0)/wavelength, Z/d, epsilonxz_interp_masked/(rhow*a**2*omega**3), levels=np.linspace(0, 0.15, 31), cmap = 'cividis')  # Line contours
contour5.set_clim(0, 0.15)  # This updates the plot AND the colorbar
ax5.plot((X_surface-x_0)/wavelength, Z_surface/d, color='#866E65')
ax5.set_xlim(0,7)
ax5.set_ylim(0, 1.15)
ax5.set_xticks([0, 2, 4, 6])
ax5.set_xlabel("$x/\lambda$")
ax5.set_ylabel("$z/d$")

# Get position of the main axis
pos = ax5.get_position()
dy = -0.01  # adjust as needed
ax5.set_position([pos.x0, pos.y0 + dy, pos.width, pos.height])

pos = ax5.get_position()

# Define a new axis for the colorbar just above the main plot
cbar_ax = fig.add_axes([pos.x0, pos.y1 + 0.02, pos.width, 0.02])  # [left, bottom, width, height]
cbar = plt.colorbar(contour5, cax=cbar_ax, orientation='horizontal')
ticks = np.linspace(0, 0.15, 4)  # 5 evenly spaced ticks from min to max
cbar.set_ticks(ticks)
cbar.set_label(r'$\epsilon_{xz}$', labelpad=5)  # Optional: label for the color bar
cbar.ax.xaxis.set_label_position('top')
cbar.ax.tick_params(labelsize=label_font_size)  # <<-- Set tick label font size here

for c in contour5.collections:
    c.set_rasterized(True)

"""
Viscous dissipation
"""
ax6 = fig.add_subplot(gs[1, 2])
contour6 = ax6.contourf((X-x_0)/wavelength, Z/d, epsilon_interp_masked/(rhow*a**2*omega**3), levels=np.linspace(0, 0.15, 31), cmap = 'cividis')  # Line contours
ax6.plot((X_surface-x_0)/wavelength, Z_surface/d, color='#866E65')
ax6.set_xlim(0,7)
ax6.set_ylim(0, 1.15)
ax6.set_xticks([0, 2, 4, 6])
ax6.set_xlabel("$x/\lambda$")
ax6.set_ylabel("$z/d$")

# Get position of the main axis
pos = ax6.get_position()
dy = -0.01  # adjust as needed
ax6.set_position([pos.x0, pos.y0 + dy, pos.width, pos.height])

pos = ax6.get_position()

# Define a new axis for the colorbar just above the main plot
cbar_ax = fig.add_axes([pos.x0, pos.y1 + 0.02, pos.width, 0.02])  # [left, bottom, width, height]
cbar = plt.colorbar(contour6, cax=cbar_ax, orientation='horizontal')
ticks = np.linspace(0, 0.15, 4)  # 5 evenly spaced ticks from min to max
cbar.set_ticks(ticks)
cbar.set_label(r'$\epsilon$', labelpad=5)  # Optional: label for the color bar
cbar.ax.xaxis.set_label_position('top')
cbar.ax.tick_params(labelsize=label_font_size)  # <<-- Set tick label font size here

for c in contour6.collections:
    c.set_rasterized(True)

"""
Convective term
"""
ax7 = fig.add_subplot(gs[2, 0])
contour7 = ax7.contourf((X-x_0)/wavelength, Z/d, convection_interp_masked/(rhow*a**2*omega**3), levels=np.linspace(-0.0001, 0.0005, 31), cmap = 'plasma')  # Line contours

# Get position of the main axis
pos = ax7.get_position()
dy = -0.025  # adjust as needed
ax7.set_position([pos.x0, pos.y0 + dy, pos.width, pos.height])

pos = ax7.get_position()
# Define a new axis for the colorbar just above the main plot
cbar_ax = fig.add_axes([pos.x0, pos.y1 + 0.02, pos.width, 0.02])  # [left, bottom, width, height]

cbar = plt.colorbar(contour7, cax=cbar_ax, orientation='horizontal')

ticks = np.linspace(0, 0.0005, 3)  # 5 evenly spaced ticks from min to max
cbar.set_ticks(ticks)

cbar.set_label('$\overline{C_k}$', labelpad=5)  # Optional: label for the color bar
cbar.ax.xaxis.set_label_position('top')
cbar.ax.tick_params(labelsize=label_font_size)  # <<-- Set tick label font size here

ax7.plot((X_mean-x_0)/wavelength, Z_mean/d, color='#866E65')
ax7.set_xlim(0,7)
ax7.set_ylim(0, 1.15)
ax7.set_xticks([0, 2, 4, 6])
ax7.set_ylabel("$z/d$")



for c in contour7.collections:
    c.set_rasterized(True)
    
"""
Transport of pressure
"""
ax8 = fig.add_subplot(gs[2, 1])
contour8 = ax8.contourf((X-x_0)/wavelength, Z/d, pressureT_interp_masked/(rhow*a**2*omega**3), levels=np.linspace(0, 0.03, 31), cmap = 'plasma')  # Line contours

# Get position of the main axis
pos = ax8.get_position()
dy = -0.025  # adjust as needed
ax8.set_position([pos.x0, pos.y0 + dy, pos.width, pos.height])

pos = ax8.get_position()

# Define a new axis for the colorbar just above the main plot
cbar_ax = fig.add_axes([pos.x0, pos.y1 + 0.02, pos.width, 0.02])  # [left, bottom, width, height]

cbar = plt.colorbar(contour8, cax=cbar_ax, orientation='horizontal')

ticks = np.linspace(0, 0.03, 4)  # 5 evenly spaced ticks from min to max
cbar.set_ticks(ticks)

cbar.set_label(r'$\overline{T_p}$', labelpad=5)  # Optional: label for the color bar
cbar.ax.xaxis.set_label_position('top')
cbar.ax.tick_params(labelsize=label_font_size)  # <<-- Set tick label font size here

ax8.plot((X_mean-x_0)/wavelength, Z_mean/d, color='#866E65')
ax8.set_xlim(0,7)
ax8.set_ylim(0, 1.15)
ax8.set_xticks([0, 2, 4, 6])
ax8.set_ylabel("$z/d$")

for c in contour8.collections:
    c.set_rasterized(True)

"""
Time-averaged Viscous diffusion
"""
ax9 = fig.add_subplot(gs[3, 0])
contour9 = ax9.contourf((X-x_0)/wavelength, Z/d, viscousDif_interp_masked/(rhow*a**2*omega**3), levels = 30, cmap = 'plasma')  # Line contours

# Get position of the main axis
pos = ax9.get_position()

# Define a new axis for the colorbar just above the main plot
cbar_ax = fig.add_axes([pos.x0, pos.y1 + 0.02, pos.width, 0.02])  # [left, bottom, width, height]

cbar = plt.colorbar(contour9, cax=cbar_ax, orientation='horizontal')
cbar.set_label(r'$\overline{D_v}$', labelpad=5)  # Optional: label for the color bar
cbar.ax.xaxis.set_label_position('top')
cbar.ax.tick_params(labelsize=label_font_size)  # <<-- Set tick label font size here

ticks = np.linspace(-0.02, 0.08, 6)  # 5 evenly spaced ticks from min to max
cbar.set_ticks(ticks)

ax9.plot((X_mean-x_0)/wavelength, Z_mean/d, color='#866E65')
ax9.set_xlim(0,7)
ax9.set_ylim(0, 1.15)
ax9.set_xticks([0, 2, 4, 6])
ax9.set_xlabel("$x/\lambda$")
ax9.set_ylabel("$z/d$")

for c in contour9.collections:
    c.set_rasterized(True)
    
"""
Time-averaged Viscous dissipation rate
"""
ax10 = fig.add_subplot(gs[3, 1])
contour10 = ax10.contourf((X-x_0)/wavelength, Z/d, viscousDis_interp_masked/(rhow*a**2*omega**3), levels=np.linspace(0, 0.1, 31), linewidths=0, antialiased=False, cmap = 'plasma')  # Line contours

# Get position of the main axis
pos = ax10.get_position()

# Define a new axis for the colorbar just above the main plot
cbar_ax = fig.add_axes([pos.x0, pos.y1 + 0.02, pos.width, 0.02])  # [left, bottom, width, height]

cbar = plt.colorbar(contour10, cax=cbar_ax, orientation='horizontal')
cbar.set_label(r'$\overline{\epsilon}$')  # Optional: label for the color bar
cbar.ax.xaxis.set_label_position('top')
cbar.ax.tick_params(labelsize=label_font_size)  # <<-- Set tick label font size here

ax10.plot((X_mean-x_0)/wavelength, Z_mean/d, color='#866E65')
ax10.set_xlim(0,7)
ax10.set_ylim(0, 1.15)
ax10.set_xticks([0, 2, 4, 6])
ax10.set_xlabel("$x/\lambda$")
ax10.set_ylabel("$z/d$")

ticks = np.linspace(0, 0.1, 6)  # 5 evenly spaced ticks from min to max
cbar.set_ticks(ticks)

for c in contour10.collections:
    c.set_rasterized(True)

"""
Depth-integrated Time-averaged Viscous dissipation
"""
ax11 = fig.add_subplot(gs[2:4, 2])

# Get position of the main axis
pos = ax11.get_position()
dy = -0.025  # adjust as needed
ax11.set_position([pos.x0, pos.y0, pos.width, pos.height+dy])

ax11.plot(x_Case2B, epsilonD_Case2B, color='black', label = 'Case 2B')
ax11.plot(x_Case2C, epsilonD_Case2C, color='#DB494E', linestyle = '--', label = 'Case 2C')
ax11.plot(x_Case4C, epsilonD_Case4C, color='#1f77b4', linestyle = '-.', label = 'Case 4C')
ax11.legend(frameon=False, fontsize=label_font_size)
ax11.set_xlim(0,7)
ax11.set_ylim(0.04, 0.1)
ax11.set_xticks([0, 2, 4, 6])
ax11.set_yticks(np.linspace(0.04, 0.1, 4))
sns.despine(ax=ax11)
ax11.set_xlabel("$x/\lambda$")
ax11.set_ylabel("$\epsilon_D/E_0 c_g$")

axes = [ax1, ax2, ax4, ax7, ax11]
labels = list("abcde")
    
for ax, label in zip(axes, labels):
    # Convert the top-left of each axis to figure coordinates
    inv = fig.transFigure.inverted()
    x_fig, y_fig = inv.transform(ax.transAxes.transform((0, 1)))
    
    # Apply a small offset
    fig.text(x_fig - 0.0425, y_fig+0.01, f"{label}", fontsize=15, fontweight='bold')
    
# plt.savefig("Figure4.pdf", dpi=1200, bbox_inches='tight', pad_inches=0, facecolor='white')
plt.show()
