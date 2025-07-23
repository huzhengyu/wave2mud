# -*- coding: utf-8 -*-
"""
Created on Wed May 21 08:39:52 2025

@author: HZY
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.gridspec as gridspec

# === Solve dispersion relation to get wave numbers k ===
g = 9.81
def get_cg(f, h):
    omega = 2*np.pi*f
    # Iterative solution for k using dispersion relation
    k = omega**2 / g  # initial guess (deep water)
    for _ in range(100):
        k = omega**2 / (g * np.tanh(k * h))
    n = 0.5 * (1 + (2 * k * h) / np.sinh(2 * k * h))
    c = omega / k
    c_g = c * n
    return c_g

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

# Example user-defined land boundary (polygon) points
Obs = pd.read_excel('../data/SourceData.xlsx',sheet_name='Sheremet2011')
Robillard2023 = pd.read_excel('../data/SourceData.xlsx',sheet_name='Robillard2023')
Num = pd.read_excel('../data/SourceData.xlsx',sheet_name='Figure5')

# Map
land_x = Obs["longitude"]
land_y = Obs["latitude"]
land_x1 = Obs["longitude1"]
land_y1 = Obs["latitude1"]
land_x2 = Obs["longitude2"]
land_y2 = Obs["latitude2"]
mud_x = Obs["mud_longitude"]
mud_y = Obs["mud_latitude"]
sand_x = Obs["sand_longitude"]
sand_y = Obs["sand_latitude"]
relictMud_x = Obs["relictMud_longitude"]
relictMud_y = Obs["relictMud_latitude"]
x_20 = Obs["20_longitude"]
y_20 = Obs["20_latitude"]
x_15 = Obs["15_longitude"]
y_15 = Obs["15_latitude"]
x_10 = Obs["10_longitude"]
y_10 = Obs["10_latitude"]
x_5_1 = Obs["5_longitude1"]
y_5_1 = Obs["5_latitude1"]
x_5_2 = Obs["5_longitude2"]
y_5_2 = Obs["5_latitude2"]
x_5_3 = Obs["5_longitude3"]
y_5_3 = Obs["5_latitude3"]
x_5_4 = Obs["5_longitude4"]
y_5_4 = Obs["5_latitude4"]

#Mean water depth
depth_x = Obs["depth_x"]
depth_z = Obs["depth_z"]

#Measured viscosity
gamma_measured = Robillard2023["gamma_Robillard2023"]
mu_measured = Robillard2023["mu_Robillard2023"] 

#wave spectra
f = Num["f"]
S_10Mar1500UT_T1 = Num["S_10Mar1500UT_T1"]
S_10Mar2100UT_T1 = Num["S_10Mar2100UT_T1"]
S_11Mar1100UT_T1 = Num["S_11Mar1100UT_T1"]
f_T2 = Num["f2"]
S_10Mar1500UT_T2 = Num["S_10Mar1500UT_T2"]
S_10Mar2100UT_T2 = Num["S_10Mar2100UT_T2"]
S_11Mar1100UT_T2 = Num["S_11Mar1100UT_T2"]

f_10Mar1500UT_T1_Obs = Obs["f_10Mar1500UT_T1"]
f_10Mar2100UT_T1_Obs = Obs["f_10Mar2100UT_T1"]
f_11Mar1100UT_T1_Obs = Obs["f_11Mar1100UT_T1"]
S_10Mar1500UT_T1_Obs = Obs["S_10Mar1500UT_T1"]
S_10Mar2100UT_T1_Obs = Obs["S_10Mar2100UT_T1"]
S_11Mar1100UT_T1_Obs = Obs["S_11Mar1100UT_T1"]
f1 = Obs["f1"]
S_10Mar1500UT_T2_Obs = Obs["S_10Mar1500UT_T2"]
S_10Mar2100UT_T2_Obs = Obs["S_10Mar2100UT_T2"]
S_11Mar1100UT_T2_Obs = Obs["S_11Mar1100UT_T2"]

fig = plt.figure(figsize=(16, 11.2))
gs = gridspec.GridSpec(4, 6, figure=fig, wspace=0.6, hspace=0.5)

# Figure a
ax1 = fig.add_subplot(gs[0:2, 0:3])

ax1.fill([-92.1733, -92.1127, -90.9112, -90.9101],[	29.6963, 29.2596, 29.2279, 29.6957], color='#E8E4E0', label='Shelly mud')
ax1.fill(relictMud_x, relictMud_y, color='#D9CCBF', label='Relict mud')                # filled area

ax1.fill(sand_x, sand_y, color='#F5EDC8', label='Sand')                # filled area
ax1.fill(mud_x, mud_y, color='#AE9C93', label='Mud')                # filled area

ax1.plot(land_x, land_y, 'k-', linewidth=1.5)  # line
ax1.fill(land_x, land_y, color='#E0E0E0')                # filled area

ax1.plot(land_x1, land_y1, 'k-', linewidth=1.5)  # line
ax1.fill(land_x1, land_y1, color='#E0E0E0')                # filled area

ax1.plot(land_x2, land_y2, 'k-', linewidth=1.5)  # line
ax1.fill(land_x2, land_y2, color='#E0E0E0')                # filled area

ax1.plot(x_20, y_20,'#006D77', linewidth=1)
ax1.text(-91.42, 28.8, '20 m', color='#006D77', fontsize=10, ha='center', va='bottom')

ax1.plot(x_15, y_15,'#006D77', linewidth=1)
ax1.text(-91.32, 28.83, '15 m', color='#006D77', fontsize=10, ha='center', va='bottom')

ax1.plot(x_10, y_10,'#006D77', linewidth=1)
ax1.text(-91.46, 29.02, '10 m', color='#006D77', fontsize=10, ha='center', va='bottom')

ax1.plot(x_5_1, y_5_1,'#006D77', linewidth=1)
ax1.text(-92.2, 29.25, '5 m', color='#006D77', fontsize=10, ha='center', va='bottom')
ax1.plot(x_5_2, y_5_2,'#006D77', linewidth=1)
ax1.text(-91.7, 29.155, '5 m', color='#006D77', fontsize=10, ha='center', va='bottom')
ax1.plot(x_5_3, y_5_3,'#006D77', linewidth=1)
ax1.text(-91.05, 28.94, '5 m', color='#006D77', fontsize=10, ha='center', va='bottom')
ax1.plot(x_5_4, y_5_4,'#006D77', linewidth=1)
ax1.text(-91.8, 29.36, '5 m', color='#006D77', fontsize=10, ha='center', va='bottom')

ax1.scatter([-91.4973, -91.4648], [	29.2199, 29.2886], marker='o', s=90, color='#E76F51', edgecolor='black')
ax1.text(-91.55, 29.21, 'T1', color='k', ha='center', va='bottom', fontweight='bold')
ax1.text(-91.51, 29.3, 'T2', color='k', ha='center', va='bottom', fontweight='bold')
ax1.text(-91.873, 29.52, 'Marsh\nIsland', color='k', ha='center', va='bottom', fontweight='bold')
ax1.text(-91.43, 29.39, 'Atchafalaya\nBay', color='k', ha='center', va='bottom', fontweight='bold')
ax1.text(-91.08, 29.5, 'Atchafalaya\nRiver', color='k', ha='center', va='bottom', fontweight='bold')

ax1.set_xlim(-92.3001, -90.9144)
ax1.set_ylim(28.76, 29.6939)

# Format ticks with degree symbols and direction
xticks = [-92, -91.5, -91]
yticks = [28.8, 29.2, 29.6]
ax1.set_xticks(xticks)
ax1.set_yticks(yticks)

# Format tick labels as strings
ax1.set_xticklabels([f"{abs(x):.1f}°W" for x in xticks], fontsize=12)
ax1.set_yticklabels([f"{y:.1f}°N" for y in yticks], fontsize=12)

# Remove axis labels
ax1.set_xlabel('')
ax1.set_ylabel('')

# ax1.grid(True)
ax1.legend(ncol=2, frameon=True)
ax1.set_aspect('equal', adjustable='box')  # maintain aspect ratio

# Figure b
ax2 = fig.add_subplot(gs[0, 3:6])
ax2.plot(depth_x, -depth_z, color='k', linewidth=1.5)
ax2.scatter([0.558352, 4.49428], [4.2539, 5.00733], marker='o', s=90, color='#E76F51', edgecolor='black')
ax2.text(4.3, 6, 'T1', color='k', ha='center', va='bottom', fontweight='bold')
ax2.text(0.4, 5.2, 'T2', color='k', ha='center', va='bottom', fontweight='bold')
ax2.set_xlim(0,12)
ax2.set_ylim(3,9)
ax2.invert_xaxis()
ax2.invert_yaxis()
ax2.set_xlabel('Cross-shore distance (km)')
ax2.set_ylabel('Depth (m)', labelpad=20)

# Figure c
ax3 = fig.add_subplot(gs[1, 3:6])
muInf = 1.80E-02
muE = 5.60E-01
gammaL = 0.4
sinSigma = 0.26
gamma = np.logspace(-3,2,100)
mu = (muInf*gamma+muE*gammaL)/(gamma+gammaL)
muABS = (muInf*gamma+muE*gammaL)/(gamma+gammaL*sinSigma)
ax3.scatter(gamma_measured,mu_measured, color='#1f77b4', s=30, label = 'Exp')
ax3.plot(gamma, mu, color='#1f77b4', linewidth=1.5, label = 'Loss')
ax3.plot(gamma, muABS, color='k', linestyle = '--', linewidth=1.5, label = 'Abs')


# Shade regions
ax3.axvspan(1e-3, gammaL, color='lightgray', alpha=0.3)
ax3.axvspan(gammaL, 1e2, color='lightblue', alpha=0.2)

# Region labels
ax3.text(2e-2, 0.15, 'Elastoviscous', ha='center', va='bottom', fontsize=10)
# ax3.text(1.5e-3, 0.1, '$\mu_E$', ha='center', va='bottom', fontsize=10)
# ax3.text(70, 3e-2, '$\mu_{\infty}$', ha='center', va='bottom', fontsize=10)
ax3.text(0.52, -0.13, '$\dot{\gamma}_{L}$', ha='center', va='bottom', fontsize=10, transform=ax3.transAxes)
ax3.text(20, 0.15, 'Liquefied viscoplastic', ha='center', va='bottom', fontsize=10)

ax3.legend()

ax3.set_xscale('log')
ax3.set_yscale('log')
ax3.set_xlim(1e-3,1e2)
ax3.set_ylim(1e-2,1e2)
ax3.set_xlabel("$\dot{\gamma}$ (s$^{-1}$)")
ax3.set_ylabel("Viscosity (Pa$\cdot$s)", labelpad=-1)

# Figure d e f
ax4 = fig.add_subplot(gs[2, 0:2])
h = 5

ax4.plot(f[7:], (S_10Mar1500UT_T1*get_cg(f, h))[7:], color='k', linewidth=1.5, label="Num")
ax4.scatter(f_10Mar1500UT_T1_Obs, S_10Mar1500UT_T1_Obs*get_cg(f_10Mar1500UT_T1_Obs, h), color='#DB494E', s=30, label="Obs")
ax4.set_ylim(5e-3,5)
ax4.set_title("10 March 1500 UT")
# ax4.legend()
handles, labels = ax4.get_legend_handles_labels()

# Define your desired order by index
order = [1,0]  # Change this to the order you want

# Apply reordered legend
ax4.legend([handles[i] for i in order],
           [labels[i] for i in order],
           frameon=False)
ax4.text(0.05, 0.9, 'T1', transform=ax4.transAxes, ha='left', va='top', fontweight='bold')

ax5 = fig.add_subplot(gs[3, 0:2])
ax5.scatter(f1, S_10Mar1500UT_T2_Obs, color='#DB494E', s=30)
ax5.text(0.05, 0.9, 'T2', transform=ax5.transAxes, ha='left', va='top', fontweight='bold')
ax5.plot(f_T2[5:], (S_10Mar1500UT_T2*get_cg(f_T2,h))[5:], color='k', linewidth=1.5, label="Num")
ax5.set_ylim(5e-3,5)

ax6 = fig.add_subplot(gs[2, 2:4])
ax6.plot(f[7:], (S_10Mar2100UT_T1*get_cg(f, h))[7:], color='k', linewidth=1.5, label="Num")
ax6.scatter(f_10Mar2100UT_T1_Obs, S_10Mar2100UT_T1_Obs*get_cg(f_10Mar2100UT_T1_Obs, h), color='#DB494E', s=30, label="Obs")
ax6.set_ylim(1e-2,20)
ax6.set_title("10 March 2100 UT")
# ax6.legend()
handles, labels = ax6.get_legend_handles_labels()

# Define your desired order by index
order = [1,0]  # Change this to the order you want

# Apply reordered legend
ax6.legend([handles[i] for i in order],
           [labels[i] for i in order],
           frameon=False)
ax6.text(0.05, 0.9, 'T1', transform=ax6.transAxes, ha='left', va='top', fontweight='bold')

ax7 = fig.add_subplot(gs[3, 2:4])
ax7.scatter(f1, S_10Mar2100UT_T2_Obs, color='#DB494E', s=30)
ax7.plot(f_T2[5:], (S_10Mar2100UT_T2*get_cg(f_T2,h))[5:], color='k', linewidth=1.5, label="Num")
ax7.text(0.05, 0.9, 'T2', transform=ax7.transAxes, ha='left', va='top', fontweight='bold')
ax7.set_ylim(1e-2,20)

ax8 = fig.add_subplot(gs[2, 4:6])
ax8.plot(f[7:], (S_11Mar1100UT_T1*get_cg(f, h))[7:], color='k', linewidth=1.5, label="Num")
ax8.scatter(f_11Mar1100UT_T1_Obs, S_11Mar1100UT_T1_Obs*get_cg(f_11Mar1100UT_T1_Obs, h), color='#DB494E', s=30, label="Obs")
ax8.set_ylim(2e-3,10)
ax8.set_title("11 March 1100 UT")
# ax8.legend()
handles, labels = ax8.get_legend_handles_labels()

# Define your desired order by index
order = [1,0]  # Change this to the order you want

# Apply reordered legend
ax8.legend([handles[i] for i in order],
           [labels[i] for i in order],
           frameon=False)

ax8.text(0.05, 0.9, 'T1', transform=ax8.transAxes, ha='left', va='top', fontweight='bold')

ax9 = fig.add_subplot(gs[3, 4:6])
ax9.scatter(f1, S_11Mar1100UT_T2_Obs, color='#DB494E', s=30)
ax9.plot(f_T2[5:], (S_11Mar1100UT_T2*get_cg(f_T2,h))[5:], color='k', linewidth=1.5, label="Num")
ax9.text(0.05, 0.9, 'T2', transform=ax9.transAxes, ha='left', va='top', fontweight='bold')
ax9.set_ylim(1e-3,9)

for ax in [ax4, ax5, ax6, ax7, ax8, ax9]:
    ax.set_xlim(0, 0.4)
    ax.set_yscale("log")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Energy flux (m$^3$)")

axes = [ax1, ax2, ax3]
labels = list("abc")
    
for ax, label in zip(axes, labels):
    # Convert the top-left of each axis to figure coordinates
    inv = fig.transFigure.inverted()
    x_fig, y_fig = inv.transform(ax.transAxes.transform((0, 1)))
    
    # Apply a small offset
    fig.text(x_fig - 0.03, y_fig-0.01, f"{label}", fontsize=18, fontweight='bold')
    
axes = [ax4, ax6, ax8]
labels = list("def")
for ax, label in zip(axes, labels):
    # Convert the top-left of each axis to figure coordinates
    inv = fig.transFigure.inverted()
    x_fig, y_fig = inv.transform(ax.transAxes.transform((0, 1)))
    
    # Apply a small offset
    fig.text(x_fig - 0.03, y_fig, f"{label}", fontsize=18, fontweight='bold')
    
# plt.savefig("Figure5.pdf", dpi=1200, bbox_inches='tight', pad_inches=0, facecolor='white')
plt.show()

