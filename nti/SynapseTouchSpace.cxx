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
// ================================================================

#include "SynapseTouchSpace.h"
#include <memory>

SynapseTouchSpace::SynapseTouchSpace(SynapseType type,
				     Params* params,
				     bool autapses)
  : _type(type),
    _params(*params),
    _autapses(autapses)
{
}


SynapseTouchSpace::SynapseTouchSpace(SynapseTouchSpace& synapseTouchSpace)
  : _type(synapseTouchSpace._type),
    _params(synapseTouchSpace._params),
    _autapses(synapseTouchSpace._autapses),
    _segmentDescriptor(synapseTouchSpace._segmentDescriptor)
{
}

bool SynapseTouchSpace::isInSpace(double key)
{ 
  bool rval=false;
  if (_type==ELECTRICAL) rval=_params.isElectricalSynapseTarget(key);
  else if(_type==CHEMICAL) rval=_params.isChemicalSynapseTarget(key);
  return rval;
}

bool SynapseTouchSpace::areInSpace(double key1, double key2)
{
  bool rval=false;
  if (_type==ELECTRICAL) rval=_params.isElectricalSynapseTarget(key1, key2, _autapses);
  else if (_type==CHEMICAL) rval=_params.isChemicalSynapseTarget(key1, key2, _autapses);
  return rval;
}

TouchSpace* SynapseTouchSpace::duplicate()
{
  return new SynapseTouchSpace(*this);
}

SynapseTouchSpace::~SynapseTouchSpace()
{
}
