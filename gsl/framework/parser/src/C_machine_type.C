// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "C_machine_type.h"
#include "SyntaxError.h"
#include "C_production.h"
#include "PhaseElement.h"
#include "GslContext.h"
#include "SyntaxErrorException.h"


void C_machine_type::internalExecute(GslContext *c)
{
  if (_machineName=="CPU") {
    _machineType = machineType::CPU;
  }
  else if (_machineName=="GPU") {
    _machineType = machineType::GPU;
  }
  else if (_machineName=="FPGA") {
    _machineType = machineType::FPGA;
  }
  else {
    std::string mes = "Unrecognized MachineType " + _machineName + "!";
    throwError(mes);
  }
}

C_machine_type::C_machine_type(const C_machine_type& rv)
  : C_production(rv), _machineName(rv._machineName), _machineType(rv._machineType)
{
}

C_machine_type::C_machine_type(const std::string& machineName, SyntaxError * error)
  : C_production(error), _machineName(machineName), _machineType(machineType::NOT_SET)
{
}

C_machine_type* C_machine_type::duplicate() const
{
   return new C_machine_type(*this);
}

C_machine_type::~C_machine_type()
{
}

void C_machine_type::checkChildren() 
{
} 

void C_machine_type::recursivePrint() 
{
   printErrorMessage();
} 
