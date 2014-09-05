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
