<img src="docs/wave2mud.jpg" width="800">

# wave2mud
Source code, model setup, source data, and scripts to replicate the results and generate the figures for the study:

**From Lab to Shelf: Physically Based Prediction of Wave Damping by Muddy Seafloors**  

## Overview

[wave2mud] is an open-source, high-fidelity, physics-based computational fluid dynamics (CFD) model developed to simulate wave propagation over cohesive muddy seafloors based on [OpenFOAM®]. The model directly incorporates shear-rate-dependent mud rheology derived from laboratory rheometer measurements. Unlike traditional inverse or tuned models, it captures both the internal structure and surface-level dissipation of waves propagating over mud—at laboratory and field scales—without viscosity tuning.

## Features

- Multiphase Navier–Stokes solver with air–water–mud interfaces via the VOF method.
- Non-Newtonian rheology: Shear-thinning, viscoplastic mud with yield stress (Hsu et al. 2013; Robillard et al. 2023).
- Laboratory-scale validation (kaolinite tank tests from Hsu et al. 2013).
- Field-scale simulation of wave damping during a storm on the Atchafalaya shelf (Sheremet et al. 2011).
- Full energy budget and dissipation diagnostics.

## Repository Structure
```bash
📁 /src/              # wave2mud solver (OpenFOAM-based)
📁 /cases/            # Benchmark setups (lab-scale and field-scale)
📁 /rheology/         # Mud rheology models 
📁 /figures/          # Python scripts for generating figures
📁 /data/             # Source data
📄 README.md
📄 LICENSE
```

## Getting Started

1. **Clone the repository**:
   ```bash
   git clone https://github.com/huzhengyu/wave2mud.git
   cd wave2mud
   ```
2. **Compile the wave generation toolbox [olaFlow]**:
   ```bash
   git clone https://github.com/phicau/olaFlow.git
   cd olaFlow
   ./allMake
   ```
3. **Compile the multiphase fluid solver and the rheology models**:
   ```bash
   cd ..
   cp -rf src/multiphaseInterFoam $FOAM_APP/solvers/multiphase/
   cp -rf src/viscosityModels $FOAM_SRC/transportModels/incompressible/
   cd $FOAM_APP/solvers/multiphase/multiphaseInterFoam
   chmod +x Allwmake
   ./Allwmake
   cd $FOAM_SRC/transportModels/incompressible/
   sed -i '8a viscosityModels/Robillard2009/Robillard2009.C' Make/files
   sed -i '8a viscosityModels/Hsu2013/Hsu2013.C' Make/files
   wmake libso
   ```
4. **Run the benchmark test cases**:
   ```bash
   cd cases/lab_scale/case2B/
   chmod +x Allrun
   ./Allrun
   cd cases/field_scale/10Mar1500UT_R09/
   chmod +x Allrun
   ./Allrun
   ```
5. **Reproduce figures**:
   ```bash
    cd figures/
    python FigureX.py
   ```

## Dependencies
- [OpenFOAM®] v1912 (or compatible multiphase solver)

- Wave generation toolbox [olaFlow]

- [Python] 3.11.7

- [NumPy], [Matplotlib], [Scipy], [pandas], [PIL], [sklearn], [seaborn]

## References
Hu, Z. and Li, Y.P.: From Lab to Shelf: Physically Based Prediction of Wave Damping by Muddy Seafloors. (In review, 2025)

Hsu, W.Y., Hwung, H.H., Hsu, T.J., Torres-Freyermuth, A., Yang, R.Y.: An experimental and numerical investigation on wave-mud interactions. Journal of Geophysical Research: Oceans 118(3), 1126–1141 (2013) https://doi.org/10.1002/jgrc.20103

Robillard, D.J., Mehta, A.J., Safak, I.: Comments on wave-induced behavior of a coastal mud. Coastal Engineering 186, 104400 (2023) https://doi.org/10.1016/j.coastaleng.2023.104400

Sheremet, A., Jaramillo, S., Su, S.-F., Allison, M.A., Holland, K.T.: Wave-mud interaction over the muddy Atchafalaya subaqueous clinoform, Louisiana, United States: Wave processes. Journal of Geophysical Research: Oceans 116(C6) (2011) https://doi.org/10.1029/2010JC006644

Higuera, P., Lara, J.L., Losada, I.J.: Realistic wave generation and active wave absorption for Navier–Stokes models: Application to OpenFOAM®. Coastal Engineering 71, 102–118 (2013) https://doi.org/10.1016/j.coastaleng.2012.07.002



*This offering is not approved or endorsed by OpenCFD Limited, producer and distributor of the OpenFOAM software via www.openfoam.com, and owner of the OPENFOAM® and OpenCFD® trade marks*

*OPENFOAM® is a registered trade mark of OpenCFD Limited, producer and distributor of the OpenFOAM software via www.openfoam.com*




[wave2mud]: https://github.com/huzhengyu/wave2mud
[OpenFOAM®]: http://openfoam.com
[multiphaseInterFoam]: http://openfoam.com
[olaFlow]: https://olaflow.github.io
[Python]: https://www.python.org/
[NumPy]: https://numpy.org/
[Matplotlib]: https://matplotlib.org/
[Scipy]:https://scipy.org/
[pandas]: https://pandas.pydata.org/
[PIL]: https://pypi.org/project/pillow/
[sklearn]: https://scikit-learn.org/stable/
[seaborn]: https://seaborn.pydata.org/