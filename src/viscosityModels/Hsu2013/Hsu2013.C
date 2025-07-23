/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2011-2017 OpenFOAM Foundation
    Copyright (C) 2017 OpenCFD Ltd.
-------------------------------------------------------------------------------
License
    This file is part of OpenFOAM.

    OpenFOAM is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    OpenFOAM is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    for more details.

    You should have received a copy of the GNU General Public License
    along with OpenFOAM.  If not, see <http://www.gnu.org/licenses/>.

\*---------------------------------------------------------------------------*/

#include "Hsu2013.H"
#include "addToRunTimeSelectionTable.H"
#include "surfaceFields.H"

// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * //

namespace Foam
{
namespace viscosityModels
{
    defineTypeNameAndDebug(Hsu2013, 0);

    addToRunTimeSelectionTable
    (
        viscosityModel,
        Hsu2013,
        dictionary
    );
}
}


// * * * * * * * * * * * * Protected Member Functions  * * * * * * * * * * * * //

Foam::tmp<Foam::volScalarField>
Foam::viscosityModels::Hsu2013::calcNu() const
{
    scalar phi((rho_.value()-1000)/(2500-1000));
    scalar A(59.1026*pow(phi,2.9378));
    scalar B(-1.2438*phi-0.6332);
    scalar C(12075846*pow(phi,12.5729));
    scalar D(2.9643*phi-1.1972);

    dimensionedScalar tone("tone", dimTime, 1.0);
    dimensionedScalar muw("muw", dimDynamicViscosity, 1e-3);

    tmp<volScalarField> sr(strainRate());

    return
    (
        min
        (
            nu0_,
            (
                muw * (1e3 * A*pow(tone*sr(), B+1) + 1e3 * C*pow(tone*sr(), D+1) + tone*sr())
                /(max(tone*sr(), scalar(VSMALL)))
            )/rho_
        )
    );
}


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::viscosityModels::Hsu2013::Hsu2013
(
    const word& name,
    const dictionary& viscosityProperties,
    const volVectorField& U,
    const surfaceScalarField& phi
)
:
    viscosityModel(name, viscosityProperties, U, phi),
    Hsu2013Coeffs_
    (
        viscosityProperties.optionalSubDict(typeName + "Coeffs")
    ),
    rho_("rho", dimDensity, Hsu2013Coeffs_),
    nu0_("nu0", dimViscosity, Hsu2013Coeffs_),
    nu_
    (
        IOobject
        (
            name,
            U_.time().timeName(),
            U_.db(),
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        calcNu()
    )
{}


// * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * * //

bool Foam::viscosityModels::Hsu2013::read
(
    const dictionary& viscosityProperties
)
{
    viscosityModel::read(viscosityProperties);

    Hsu2013Coeffs_ =
        viscosityProperties.optionalSubDict(typeName + "Coeffs");

    Hsu2013Coeffs_.readEntry("rho", rho_);
    Hsu2013Coeffs_.readEntry("nu0", nu0_);

    return true;
}


// ************************************************************************* //
