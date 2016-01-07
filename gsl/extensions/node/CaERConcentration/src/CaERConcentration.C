// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BMC-YKT-08-23-2011-2
//
// (C) Copyright IBM Corp. 2005-2014  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#include "Lens.h"
#include "CaERConcentration.h"
#include "CG_CaERConcentration.h"
#include "rndm.h"

void CaERConcentration::initializeCaConcentration(RNG& rng) 
{
}

void CaERConcentration::solve(RNG& rng) 
{
}

void CaERConcentration::finish(RNG& rng) 
{
}

void CaERConcentration::setReceptorCaCurrent(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_CaERConcentrationInAttrPSet* CG_inAttrPset, CG_CaERConcentrationOutAttrPSet* CG_outAttrPset) 
{
}

void CaERConcentration::setInjectedCaCurrent(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_CaERConcentrationInAttrPSet* CG_inAttrPset, CG_CaERConcentrationOutAttrPSet* CG_outAttrPset) 
{
}

void CaERConcentration::setProximalJunction(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_CaERConcentrationInAttrPSet* CG_inAttrPset, CG_CaERConcentrationOutAttrPSet* CG_outAttrPset) 
{
}

bool CaERConcentration::checkSite(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_CaERConcentrationInAttrPSet* CG_inAttrPset, CG_CaERConcentrationOutAttrPSet* CG_outAttrPset) 
{
}

bool CaERConcentration::confirmUniqueDeltaT(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_CaERConcentrationInAttrPSet* CG_inAttrPset, CG_CaERConcentrationOutAttrPSet* CG_outAttrPset) 
{
}

CaERConcentration::~CaERConcentration() 
{
}

