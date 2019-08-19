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

#include "C_machine_type.h"
#include "SyntaxError.h"
#include "C_production.h"
#include "PhaseElement.h"
#include "LensContext.h"
#include "SyntaxErrorException.h"


void C_machine_type::internalExecute(LensContext *c)
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
