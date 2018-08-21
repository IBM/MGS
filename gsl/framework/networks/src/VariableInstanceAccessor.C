// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BCM-YKT-07-18-2017
//
// (C) Copyright IBM Corp. 2005-2017  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#include "VariableInstanceAccessor.h"
//#include "CompCategory.h"
//#include "Simulation.h"

VariableInstanceAccessor::VariableInstanceAccessor()
   : _variable(0), _variableIndex(0)
{
}

Variable* VariableInstanceAccessor::getVariable()
{
  return _variable;
}

void VariableInstanceAccessor::setVariable(Variable* n)
{
  _variable=n;
}

int VariableInstanceAccessor::getVariableIndex() const {
  return _variableIndex;
}      

void VariableInstanceAccessor::setVariableIndex(int index) { 
  _variableIndex = index;
}

VariableInstanceAccessor::~VariableInstanceAccessor()
{
}
