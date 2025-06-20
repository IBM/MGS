// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
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
