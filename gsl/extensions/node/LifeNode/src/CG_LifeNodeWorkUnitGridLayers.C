// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM
//
// BCM-YKT-12-03-2018
//
//  (C) Copyright IBM Corp. 2005-2018  All rights reserved   .
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#include "Lens.h"
#include "CG_LifeNodeWorkUnitGridLayers.h"
#include "CG_LifeNodeCompCategory.h"
#include "GridLayerData.h"
#include "WorkUnit.h"
#include "rndm.h"

CG_LifeNodeWorkUnitGridLayers::CG_LifeNodeWorkUnitGridLayers(GridLayerData* arg, void (CG_LifeNodeCompCategory::*computeState) (GridLayerData*, CG_LifeNodeWorkUnitGridLayers*), CG_LifeNodeCompCategory* compCategory) 
   : WorkUnit(), _arg(arg), _compCategory(compCategory), _computeState(computeState){
   _rng.reSeed(urandom(_compCategory->getSimulation().getWorkUnitRandomSeedGenerator()), _compCategory->getSimulation().getRank());
}

void CG_LifeNodeWorkUnitGridLayers::execute() 
{
   (*_compCategory.*_computeState)(_arg, this);
}

RNG& CG_LifeNodeWorkUnitGridLayers::getRNG() 
{
   return _rng;
}

void CG_LifeNodeWorkUnitGridLayers::setGPUMachineID(int GPUMachineID) 
{
   _GPUMachineID = GPUMachineID;
}

int CG_LifeNodeWorkUnitGridLayers::getGPUMachineID() 
{
   return _GPUMachineID;
}

CG_LifeNodeWorkUnitGridLayers::~CG_LifeNodeWorkUnitGridLayers() 
{
}

