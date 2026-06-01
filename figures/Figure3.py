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

Num = pd.read_excel('../data/SourceData.xlsx',sheet_name = 'Figure3')

X_surface = Num["mud_surface_x"].dropna()
Z_surface = Num["mud_surface_z"].dropna()+0.06

X_newtonian_surface = Num["Newtonian_mud_surface_x"].dropna()
Z_newtonian_surface = Num["Newtonian_mud_surface_z"].dropna()+0.06

X_mean = np.array([5.24,13.35])
Z_mean = np.array([0,0])+0.06

strainRate = Num["strainRate_Magnitude"].dropna()
strainRate_newtonian = Num["Newtonian_strainRate_Magnitude"].dropna()

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

mu = Num["nu.mud"].dropna()*rho_b
muM = Num["nu.mudMean"].dropna()*rho_b
epsilonxx = Num["epsilon_xx"].dropna()
epsilonxz = Num["epsilon_xz"].dropna()
epsilon = Num["epsilon"].dropna()

epsilonxx_newtonian = Num["Newtonian_epsilon_xx"].dropna()
epsilonxz_newtonian = Num["Newtonian_epsilon_xz"].dropna()
epsilon_newtonian = Num["Newtonian_epsilon"].dropna()

# # Interpolate scattered data to a grid
strainRate_interp = griddata((x, z), strainRate, (X, Z), method='linear')
mu_interp = griddata((x, z), mu, (X, Z), method='linear')
muM_interp = griddata((x, z), muM, (X, Z), method='linear')
epsilonxx_interp = griddata((x, z), epsilonxx, (X, Z), method='linear')
epsilonxz_interp = griddata((x, z), epsilonxz, (X, Z), method='linear')
epsilon_interp = griddata((x, z), epsilon, (X, Z), method='linear')

strainRate_newtonian_interp = griddata((x_newtonian, z_newtonian), strainRate_newtonian, (X_newtonian, Z_newtonian), method='linear')
epsilonxx_newtonian_interp = griddata((x_newtonian, z_newtonian), epsilonxx_newtonian, (X_newtonian, Z_newtonian), method='linear')
epsilonxz_newtonian_interp = griddata((x_newtonian, z_newtonian), epsilonxz_newtonian, (X_newtonian, Z_newtonian), method='linear')
epsilon_newtonian_interp = griddata((x_newtonian, z_newtonian), epsilon_newtonian, (X_newtonian, Z_newtonian), method='linear')

# Step 2: Create a 2D mask where Z > surface elevation at each X
surface_interp = interp1d(X_surface, Z_surface, bounds_error=False, fill_value=np.nan)
Z_surf_on_grid = surface_interp(X[0, :])  # 1D slice of surface for each x in the grid

surface_newtonian_interp = interp1d(X_newtonian_surface, Z_newtonian_surface, bounds_error=False, fill_value=np.nan)
Z_surf_on_grid_newtonian = surface_newtonian_interp(X_newtonian[0, :])  # 1D slice of surface for each x in the grid

mean_interp = interp1d(X_mean, Z_mean, bounds_error=False, fill_value=np.nan)
Z_mean_on_grid = mean_interp(X[0, :])

# Step 2: Create a 2D mask where Z > surface elevation at each X
Z_surface_grid = np.tile(Z_surf_on_grid, (Z.shape[0], 1))  # repeat to shape of P_interp
mask1 = Z > Z_surface_grid  # mask values above surface

Z_mean_grid = np.tile(Z_mean_on_grid, (Z.shape[0], 1))  # repeat to shape of P_interp
mask2 = Z > Z_mean_grid  # mask values above surface

Z_surface_grid_newtonian = np.tile(Z_surf_on_grid_newtonian, (Z_newtonian.shape[0], 1))  # repeat to shape of P_interp
mask3 = Z_newtonian > Z_surface_grid_newtonian  # mask values above surface

# Step 3: Apply the mask
strainRate_interp_masked = np.ma.masked_where(mask1, strainRate_interp)
mu_interp_masked = np.ma.masked_where(mask1, mu_interp)
muM_interp_masked = np.ma.masked_where(mask2, muM_interp)
epsilonxx_interp_masked = np.ma.masked_where(mask1, epsilonxx_interp)
epsilonxz_interp_masked = np.ma.masked_where(mask1, epsilonxz_interp)
epsilon_interp_masked = np.ma.masked_where(mask1, epsilon_interp)

strainRate_newtonian_interp_masked = np.ma.masked_where(mask3, strainRate_newtonian_interp)
epsilonxx_newtonian_interp_masked = np.ma.masked_where(mask3, epsilonxx_newtonian_interp)
epsilonxz_newtonian_interp_masked = np.ma.masked_where(mask3, epsilonxz_newtonian_interp)
epsilon_newtonian_interp_masked = np.ma.masked_where(mask3, epsilon_newtonian_interp)

gamma = strainRate_interp_masked.compressed()
gamma_newtonian = strainRate_newtonian_interp_masked.compressed()   

bins = np.logspace(np.log10(gamma.min()), np.log10(gamma.max()), 100)

epsxx   = epsilonxx_interp_masked.compressed()
epsxx_newtonian   = epsilonxx_newtonian_interp_masked.compressed()

epsxz   = epsilonxz_interp_masked.compressed()
epsxz_newtonian   = epsilonxz_newtonian_interp_masked.compressed()

eps   = epsilon_interp_masked.compressed()
eps_newtonian   = epsilon_newtonian_interp_masked.compressed()

# Create a figure with subplots
fig = plt.figure(figsize=(12, 10))
gs = gridspec.GridSpec(4, 3, figure=fig, wspace=0.3, hspace=0.9)

# Plot the schematic in the first subplot
ax1 = fig.add_subplot(gs[0, 0])
contour1 = ax1.contourf((X - x_0)/wavelength, Z / d, strainRate_interp_masked/(a*omega/d), levels=np.linspace(0, 2.5, 31), cmap = cm.PuBu_r)  # Line contours cm.PuBu_r

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

# ax1.plot((Xw_surface-x_0)/wavelength, Zw_surface/d, '--', color='#003366')
ax1.plot((X_surface-x_0)/wavelength, Z_surface/d, '-', color='#866E65')
ax1.set_xlim(0,7)
ax1.set_ylim(0, 1.15)

# Apply to y-axis
# ax1.set_yticks([0, 1.0, 5.0])
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
Normal viscous dissipation XX
"""
ax4 = fig.add_subplot(gs[1, 0])
contour4 = ax4.contourf((X-x_0)/wavelength, Z/d, epsilonxx_interp_masked/(rhow*a**2*omega**3), levels=np.linspace(0, 0.005, 31), cmap = 'cividis')  # Line contours
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
Shear viscous dissipation XZ
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
Total viscous dissipation
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
Normal viscous dissipation XX distribution
"""
ax7 = fig.add_subplot(gs[2, 0])

D_bin, edges = np.histogram(gamma, bins=bins, weights=epsxx/(rhow*a**2*omega**3))
D_bin_N, _ = np.histogram(gamma_newtonian, bins=bins, weights=epsxx_newtonian/(rhow*a**2*omega**3))

widths = np.diff(edges)

D_pdf = D_bin / (D_bin.sum() * widths)
D_pdf_N = D_bin_N / (D_bin_N.sum() * widths)

centers = np.sqrt(edges[:-1] * edges[1:])

# Get position of the main axis
pos = ax7.get_position()
dy = -0.04  # adjust as needed
ax7.set_position([pos.x0, pos.y0 + dy, pos.width, pos.height+0.07])

ax7.semilogx(centers, D_pdf, 'o-', color='#1f77b4', markersize=3, label='Non-Newtonian') 
ax7.semilogx(centers, D_pdf_N, 's--', color='#DB494E', markersize=3, label='Newtonian') 
sns.despine(ax=ax7)
ax7.set_xlim([0.001, 3])
ax7.set_ylim([-0.5, 6])
ax7.set_xlabel('$|\dot{\gamma}|$') 
ax7.set_ylabel('$\epsilon_{xx}$') 

ax7.legend(frameon=False, fontsize=label_font_size, loc='upper left',bbox_to_anchor=(0, 1.05)) 
# ax7.grid(True, which='both', alpha=0.3) 

"""
Normal viscous dissipation XZ distribution
"""
ax8 = fig.add_subplot(gs[2, 1])

D_bin, edges = np.histogram(gamma, bins=bins, weights=epsxz/(rhow*a**2*omega**3))
D_bin_N, _ = np.histogram(gamma_newtonian, bins=bins, weights=epsxz_newtonian/(rhow*a**2*omega**3))

widths = np.diff(edges)

D_pdf = D_bin / (D_bin.sum() * widths)
D_pdf_N = D_bin_N / (D_bin_N.sum() * widths)

centers = np.sqrt(edges[:-1] * edges[1:])

# Get position of the main axis
pos = ax8.get_position()
dy = -0.04  # adjust as needed
ax8.set_position([pos.x0, pos.y0 + dy, pos.width, pos.height+0.07])

ax8.semilogx(centers, D_pdf, 'o-', color='#1f77b4', markersize=3, label='Non-Newtonian') 
ax8.semilogx(centers, D_pdf_N, 's--', color='#DB494E', markersize=3, label='Newtonian') 
sns.despine(ax=ax8)
ax8.set_xlim([0.001, 3])
ax8.set_ylim([-0.13, 1.5])
ax8.set_xlabel('$|\dot{\gamma}|$') 
ax8.set_ylabel('$\epsilon_{xz}$') 

ax8.legend(frameon=False, fontsize=label_font_size, loc='upper left',bbox_to_anchor=(0, 1.05)) 

"""
Total viscous dissipation distribution
"""
ax9 = fig.add_subplot(gs[2, 2])

D_bin, edges = np.histogram(gamma, bins=bins, weights=eps/(rhow*a**2*omega**3))
D_bin_N, _ = np.histogram(gamma_newtonian, bins=bins, weights=eps_newtonian/(rhow*a**2*omega**3))

widths = np.diff(edges)

D_pdf = D_bin / (D_bin.sum() * widths)
D_pdf_N = D_bin_N / (D_bin_N.sum() * widths)

centers = np.sqrt(edges[:-1] * edges[1:])

# Get position of the main axis
pos = ax9.get_position()
dy = -0.04  # adjust as needed
ax9.set_position([pos.x0, pos.y0 + dy, pos.width, pos.height+0.07])

ax9.semilogx(centers, D_pdf, 'o-', color='#1f77b4', markersize=3, label='Non-Newtonian') 
ax9.semilogx(centers, D_pdf_N, 's--', color='#DB494E', markersize=3, label='Newtonian') 
sns.despine(ax=ax9)
ax9.set_xlim([0.001, 3])
ax9.set_ylim([-0.15, 1.8])
ax9.set_xlabel('$|\dot{\gamma}|$') 
ax9.set_ylabel('$\epsilon$') 

ax9.legend(frameon=False, fontsize=label_font_size, loc='upper left',bbox_to_anchor=(0, 1.05)) 
    
    
axes = [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9]
labels = list("abcdefghi")
    
for ax, label in zip(axes, labels):
    # Convert the top-left of each axis to figure coordinates
    inv = fig.transFigure.inverted()
    x_fig, y_fig = inv.transform(ax.transAxes.transform((0, 1)))
    
    # Apply a small offset
    if ax != ax7 and ax != ax8 and ax != ax9:
        fig.text(x_fig - 0.0425, y_fig+0.01, f"{label}", fontsize=15, fontweight='bold')
    else:
        fig.text(x_fig - 0.0425, y_fig-0.01, f"{label}", fontsize=15, fontweight='bold')
 
plt.savefig("Figure3.pdf", dpi=1200, bbox_inches='tight', pad_inches=0, facecolor='white')
plt.show()
