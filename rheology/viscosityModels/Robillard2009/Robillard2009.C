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

#include "Robillard2009.H"
#include "addToRunTimeSelectionTable.H"
#include "surfaceFields.H"

// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * //

namespace Foam
{
namespace viscosityModels
{
    defineTypeNameAndDebug(Robillard2009, 0);

    addToRunTimeSelectionTable
    (
        viscosityModel,
        Robillard2009,
        dictionary
    );
}
}


// * * * * * * * * * * * * Protected Member Functions  * * * * * * * * * * * * //

Foam::tmp<Foam::volScalarField>
Foam::viscosityModels::Robillard2009::calcNu() const
{
    tmp<volScalarField> sr(strainRate());

    return
    (
        (nuInf_*sr() + nuE_*gammaL_)/(sr()+gammaL_)
    );
}


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::viscosityModels::Robillard2009::Robillard2009
(
    const word& name,
    const dictionary& viscosityProperties,
    const volVectorField& U,
    const surfaceScalarField& phi
)
:
    viscosityModel(name, viscosityProperties, U, phi),
    Robillard2009Coeffs_
    (
        viscosityProperties.optionalSubDict(typeName + "Coeffs")
    ),
    nuInf_("nuInf", dimViscosity, Robillard2009Coeffs_),
    nuE_("nuE", dimViscosity, Robillard2009Coeffs_),
    gammaL_("gammaL", dimless/dimTime, Robillard2009Coeffs_),
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

bool Foam::viscosityModels::Robillard2009::read
(
    const dictionary& viscosityProperties
)
{
    viscosityModel::read(viscosityProperties);

    Robillard2009Coeffs_ =
        viscosityProperties.optionalSubDict(typeName + "Coeffs");

    Robillard2009Coeffs_.readEntry("nuInf", nuInf_);
    Robillard2009Coeffs_.readEntry("nuE", nuE_);
    Robillard2009Coeffs_.readEntry("gammaL", gammaL_);

    return true;
}


// ************************************************************************* //
