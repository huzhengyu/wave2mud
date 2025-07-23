import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
import seaborn as sns
import matplotlib.gridspec as gridspec

# Nature-style settings
sns.set_theme(style='white')  # base white background
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

label_font_size = 12
legend_font_size = 9

# Load data
Num = pd.read_excel('../data/SourceData.xlsx',sheet_name = 'Figure3')
Exp = pd.read_excel('../data/SourceData.xlsx',sheet_name = 'Hsu2013') 

# Experimental
t1, eta1 = Exp['t_x6.5m_Hsu13'].dropna(), Exp['eta_x6.5m_Hsu13'].dropna()
t2, eta2 = Exp['t_x9.76m_Hsu13'].dropna(), Exp['eta_x9.76m_Hsu13'].dropna()
t3, eta3 = Exp['t_x11.93m_Hsu13'].dropna(), Exp['eta_x11.93m_Hsu13'].dropna()

# Numerical
t_modeled = Num['t_Num'].dropna()
eta1_modeled = Num['eta_x6.5m_Num'].dropna()
eta2_modeled = Num['eta_x9.76m_Num'].dropna()
eta3_modeled = Num['eta_x11.93m_Num'].dropna()

x = Exp['x_Hsu13'].dropna()
Hx_Case2C = Exp['Hx_CASE2C_Hsu13'].dropna()
error1_Hx_Case2C = Exp['ERR_CASE2C_Hsu13'].dropna()
Hx_CASE4C = Exp['Hx_CASE4C_Hsu13'].dropna()
error2_Hx_CASE4C = Exp['ERR_CASE4C_Hsu13'].dropna()

x_modeled = Num['x_Num'].dropna()
Hx_Case2C_modeled = Num['Hx_CASE2C_Num'].dropna()
Hx_CASE4C_modeled = Num['Hx_CASE4C_Num'].dropna()

ki_measured = Num['ki_measured'].dropna()
predicted = Num['ki_predicted'].dropna()
error = Num['uncertainties'].dropna()
outlier_indices = [16, 21]
main_indices = [i for i in range(len(ki_measured)) if i not in outlier_indices]
r2 = r2_score(ki_measured[main_indices], predicted[main_indices])

measured_hsu = Exp['ki_measured_Hsu13NM'].dropna()
predicted_hsu = Exp['ki_predicted_Hsu13NM'].dropna()
r2_Hsu13 = r2_score(measured_hsu, predicted_hsu)

U1_Case2B = Exp['U_crest_CASE2B_Hsu13'].dropna()
U1Err_Case2B = Exp['Error_U_crest_CASE2B_Hsu13'].dropna()
z = Exp['z_Hsu13'].dropna()
U2_Case2B = Exp['U_reversal_CASE2B_Hsu13'].dropna()
U2Err_Case2B = Exp['Error_U_reversal_CASE2B_Hsu13'].dropna()
theta_Case2B = Exp['theta_CASE2B_Hsu13'].dropna()

U1_Case2B_model = Num['U_crest_CASE2B_Num'].dropna()
U2_Case2B_model = Num['U_reversal_CASE2B_Num'].dropna()
z_model = Num['z_Num'].dropna()
theta_Case2B_model = Num['theta_CASE2B_Num'].dropna()

t_U = Exp['t_U_CASE2B_Hsu13'].dropna()
u_Case2B = Exp['U_CASE2B_Hsu13'].dropna()
t_gamma = Exp['t_gamma_CASE2B_Hsu13'].dropna()
gamma_Case2B = Exp['gamma_CASE2B_Hsu13'].dropna()
t_mu = Exp['t_mu_CASE2B_Hsu13'].dropna()
mu_Case2B = Exp['mu_CASE2B_Hsu13'].dropna()

t_gamma_model = Num['t_CASE2B_Num'].dropna()
u_Case2B_model = Num['u_CASE2B_Num'].dropna()
gamma_Case2B_model = Num['gamma_CASE2B_Num'].dropna()
mu_Case2B_model = Num['mu_CASE2B_Num'].dropna()

# Setup figure
fig = plt.figure(figsize=(14.2, 9))
gs = gridspec.GridSpec(6, 9, figure=fig, wspace=1.8, hspace=1.2)

"""
Figure 2a
"""
# Define styles
scatter_kwargs = dict(facecolors='none', edgecolors='#0072B2', s=6, zorder=3, label='Exp')
line_kwargs = dict(color='black', linestyle='--', linewidth=1.5, zorder=2, label='Num')

ax1 = fig.add_subplot(gs[0, 0:3])
# Plot each subplot
ax1.scatter(t1, eta1, **scatter_kwargs)
ax1.plot(t_modeled - 7.72, eta1_modeled, **line_kwargs)
ax1.set_ylabel(r"$\eta$ (m)", fontsize = label_font_size)

ax2 = fig.add_subplot(gs[1, 0:3])
ax2.scatter(t2, eta2, **scatter_kwargs)
ax2.plot(t_modeled - 11.35, eta2_modeled, **line_kwargs)
ax2.set_ylabel(r"$\eta$ (m)", fontsize = label_font_size)

ax3 = fig.add_subplot(gs[2, 0:3])
ax3.scatter(t3, eta3, **scatter_kwargs)
ax3.plot(t_modeled - 12.6, eta3_modeled, **line_kwargs)
ax3.set_ylabel(r"$\eta$ (m)", fontsize = label_font_size)
ax3.set_xlabel(r"$t$ (s)", fontsize = label_font_size)

# Add legend only to the last subplot
ax3.legend(fontsize=legend_font_size, loc='upper right',bbox_to_anchor=(1.05, 1.3), ncol=2)

# Customize axes
locs = ['6.5','9.76','11.93']
for ax, loc in zip([ax1, ax2, ax3], locs):
    ax.set_xlim([0, 10])
    ax.set_ylim([-0.07, 0.07])
    ax.set_yticks([-0.05, 0, 0.05])
    ax.text(0.03, 0.95, f"$x$ = {loc} m", transform=ax.transAxes, fontsize=10)

"""
Figure 2b
"""
ax4 = fig.add_subplot(gs[0:3, 3:6])
# Plot measured data with error bars
ax4.errorbar(x, Hx_Case2C, yerr=error1_Hx_Case2C, fmt='o', markersize=6, color='#1f77b4', elinewidth=1, capsize=2, label='Case 2C')
ax4.errorbar(x, Hx_CASE4C, yerr=error2_Hx_CASE4C, fmt='s', markersize=6, color='#DB494E', elinewidth=1, capsize=2, label='Case 4C')

# Plot modeled data
ax4.plot(x_modeled, Hx_Case2C_modeled, linestyle='--', color='k', linewidth=1.5, label='Num')
ax4.plot(x_modeled, Hx_CASE4C_modeled, linestyle='--', color='k', linewidth=1.5)

# Labels and legend
ax4.set_xlabel("$x$ (m)", fontsize=label_font_size)
ax4.set_ylabel("$H_x$ (m)", fontsize=label_font_size)
# ax4.legend(frameon=False, fontsize=legend_font_size)
# ax8.legend(frameon=False, fontsize=legend_font_size)
handles, labels = ax4.get_legend_handles_labels()

# Define your desired order by index
order = [1,2,0]  # Change this to the order you want

# Apply reordered legend
ax4.legend([handles[i] for i in order],
           [labels[i] for i in order],
           frameon=False,
           fontsize=legend_font_size)

# Set limits
ax4.set_xlim(min(x_modeled), max(x_modeled))
ax4.set_ylim(0.02, 0.1)
ax4.set_yticks(np.linspace(0.02, 0.1, 5))

"""
Figure 2c
"""
ax5 = fig.add_subplot(gs[0:3, 6:9])
# Hsu13
ax5.plot(measured_hsu, predicted_hsu, marker='s', markersize=6, linestyle='None', color='black', label=f'Hsu13 NM, $R^2$ = {r2_Hsu13:.2f}')

# Prediction with error bars
ax5.errorbar(ki_measured[main_indices], predicted[main_indices], yerr=error[main_indices],
            fmt='o', markersize=6, color='#1f77b4', ecolor='lightgray', elinewidth=1, capsize=2,
            label=f'This study,  $R^2$ =  {r2:.2f}')

# Outliers
ax5.errorbar(ki_measured[outlier_indices], predicted[outlier_indices], yerr=error[outlier_indices],
            fmt='o', markersize=6, color='#DB494E', ecolor='lightgray', elinewidth=1, capsize=2)

# Annotations with arrows
ax5.annotate("Case 3F",
            xy=(ki_measured[16], predicted[16]),
            xytext=(0.06, 0.055),
            textcoords='data',
            arrowprops=dict(arrowstyle='->', color='#DB494E', lw=0.8),
            color='#DB494E', fontsize=legend_font_size)

ax5.annotate("Case 4F",
            xy=(ki_measured[21], predicted[21]),
            xytext=(0.05, 0.035),
            textcoords='data',
            arrowprops=dict(arrowstyle='->', color='#DB494E', lw=0.8),
            color='#DB494E', fontsize=legend_font_size)

# 1:1 reference line
ax5.plot([0.015, 0.075], [0.015, 0.075], linestyle='--', color='gray', linewidth=1.5)

# Axes labels
ax5.set_xlabel("Measured $k_i$ (m$^{-1}$)", fontsize=label_font_size)
ax5.set_ylabel("Predicted $k_i$ (m$^{-1}$)", fontsize=label_font_size)

# Set equal aspect and limits
ax5.set_xlim(0.015, 0.075)
ax5.set_ylim(0.015, 0.096)
ax5.set_aspect((0.075-0.015)/(0.096-0.015), adjustable='box')
ax5.set_yticks(np.linspace(0.02, 0.08, 4))

# Legend
ax5.legend(frameon=False, fontsize=legend_font_size, loc='lower right')

"""
Figure 2d
"""
ax6 = fig.add_subplot(gs[3:6, 0:2])

# Measurement with error bars
ax6.errorbar(U1_Case2B, z, xerr=U1Err_Case2B,
            fmt='o', markersize=6, color='#1f77b4', ecolor='gray', elinewidth=1, capsize=2,
            label='Exp')
ax6.plot(U1_Case2B_model, z_model, **line_kwargs)
ax6.text(-0.03, 0.025, "Mud", fontsize=legend_font_size, color='black', va='bottom')
ax6.text(-0.03, 0.155, "Water", fontsize=legend_font_size, color='black', va='top')
ax6.set_xlim(-0.05, 0.15)
ax6.set_xticks([0, 0.15])
ax6.set_xlabel("$U_x$ (m/s)", fontsize=label_font_size)

ax7 = fig.add_subplot(gs[3:6, 2:4])
# Measurement with error bars
ax7.errorbar(U2_Case2B, z, xerr=U2Err_Case2B,
            fmt='o', markersize=6, color='#1f77b4', ecolor='gray', elinewidth=1, capsize=2,
            label='Exp')
ax7.plot(U2_Case2B_model, z_model, **line_kwargs)
ax7.set_xlim(-0.05, 0.15)
ax7.set_xticks([0, 0.15])
ax7.set_xlabel("$U_x$ (m/s)", fontsize=label_font_size)

ax8 = fig.add_subplot(gs[3:6, 4:6])
ax8.errorbar(theta_Case2B, z, xerr=0,
            fmt='o', markersize=6, color='#1f77b4', ecolor='gray', elinewidth=1, capsize=2,
            label='Exp')
ax8.plot(theta_Case2B_model, z_model[1:len(theta_Case2B_model)+1], **line_kwargs)
ax8.set_xlim(-50, 10)
ax8.set_xlabel(r"$\theta\ (^\circ)$")

# ax8.legend(frameon=False, fontsize=legend_font_size)
handles, labels = ax8.get_legend_handles_labels()

# Define your desired order by index
order = [1,0]  # Change this to the order you want

# Apply reordered legend
ax8.legend([handles[i] for i in order],
           [labels[i] for i in order],
           frameon=False,
           fontsize=legend_font_size)

# Customize axes
for ax in [ax6, ax7, ax8]:
    ax.set_ylim(0, 0.25)
    ax.set_ylabel("$z$ (m)", fontsize=label_font_size)
    # Water phase: from y=0.06 to 0.25
    ax.axhspan(0.06, 0.25, xmin=0.0, xmax=1.0, facecolor='#a6bddb', alpha=0.3, label='Water phase')
    
    # Mud phase: from y=0.00 to 0.06
    ax.axhspan(0.00, 0.06, xmin=0.0, xmax=1.0, facecolor='#d9d9d9', alpha=0.4, label='Mud phase')
    
    ax.tick_params(axis='both', which='both',
               direction='in',
               bottom=True, left=True, 
               width=0.8, length=4, color='black',labelsize=label_font_size)
    
# # Middle bottom row
ax9 = fig.add_subplot(gs[3, 6:9])
ax9.scatter(t_U/0.9, u_Case2B,**scatter_kwargs)
ax9.plot(t_gamma_model/0.9, u_Case2B_model, **line_kwargs)
ax9.set_xlim(0, 1.01)
ax9.set_ylim(-0.19, 0.2)
ax9.set_xlabel("$t/T$")
ax9.set_ylabel(r"$u$ (m/s)")
ax9.legend(frameon=False, fontsize=legend_font_size, loc='upper right',bbox_to_anchor=(1.03, 1.25), ncol=2)

ax10 = fig.add_subplot(gs[4, 6:9])
ax10.scatter(t_gamma/0.9, gamma_Case2B,**scatter_kwargs)
ax10.plot(t_gamma_model/0.9, gamma_Case2B_model, **line_kwargs)
ax10.set_xlim(0, 1.01)
ax10.set_xlabel("$t/T$")
ax10.set_ylabel("$|\dot{\gamma}|$ (s$^{-1}$)")
ax10.set_ylim(0.01, 0.6)
ax10.set_yticks([0.3,0.6])


ax11 = fig.add_subplot(gs[5, 6:9])
ax11.scatter(t_mu/0.9, mu_Case2B,**scatter_kwargs)
ax11.plot(t_gamma_model/0.9, mu_Case2B_model, **line_kwargs)
ax11.set_xlim(0, 1.01)
ax11.set_ylim(1, 40)
ax11.set_xlabel("$t/T$")
ax11.set_ylabel("$\mu$ (Pa$\cdot$s)")

axes = [ax1, ax4,ax5, ax6, ax9]
labels = list("abcde")

# Customize axes
for ax in [ax1, ax2, ax3, ax4, ax5, ax9, ax10, ax11]:
    sns.despine(ax=ax)
    ax.tick_params(axis='both', which='both',
               direction='in',
               bottom=True, left=True, 
               width=0.8, length=4, color='black',labelsize=label_font_size)
    
for ax, label in zip(axes, labels):
    # Convert the top-left of each axis to figure coordinates
    inv = fig.transFigure.inverted()
    x_fig, y_fig = inv.transform(ax.transAxes.transform((0, 1)))
    
    # Apply a small offset
    fig.text(x_fig - 0.05, y_fig, f"{label}", fontsize=label_font_size+2, fontweight='bold')

# Final layout
# plt.savefig("Figure3.pdf", dpi=1200, bbox_inches='tight', pad_inches=0)
plt.show()