// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
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

bool SynapseTouchSpace::isInSpace(key_size_t key)
{ 
  bool rval=false;
  if (_type==ELECTRICAL) rval=_params.isElectricalSynapseTarget(key);
  else if(_type==CHEMICAL) rval=_params.isChemicalSynapseTarget(key);
  return rval;
}

bool SynapseTouchSpace::areInSpace(key_size_t key1, key_size_t key2)
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
