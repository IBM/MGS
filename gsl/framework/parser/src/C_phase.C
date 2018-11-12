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

#include "C_phase.h"
#include "SyntaxError.h"
#include "C_production.h"
#include "PhaseDataItem.h"
#include "Phase.h"
#include "LensContext.h"
#include "SyntaxErrorException.h"


void C_phase::internalExecute(LensContext *c)
{
   std::auto_ptr<Phase> dup;
   c->getCurrentPhase(dup);
   dup->setName(_phase);
   if (_machineType) {
     _machineType->execute(c);
     dup->setMachineType(_machineType->getMachineType());
   }
   Phase* insToSim = dup.get();
   std::auto_ptr<DataItem> pdi(new PhaseDataItem(dup));
   try {
      c->symTable.addEntry(_phase, pdi);
   } catch (SyntaxErrorException& e) {
      throwError("While adding phase, " + e.getError());
   }
   insToSim->addToSimulation(c->sim);
}

C_phase::C_phase(const C_phase& rv)
  : C_production(rv), _phase(rv._phase), _machineType(0)
{
  if (rv._machineType) {
    _machineType = rv. _machineType->duplicate();
  }
}

C_phase::C_phase(const std::string& phase, C_machine_type* mType, SyntaxError * error)
  : C_production(error), _phase(phase), _machineType(mType)
{
}

C_phase* C_phase::duplicate() const
{
   return new C_phase(*this);
}

C_phase::~C_phase()
{
}

void C_phase::checkChildren() 
{
} 

void C_phase::recursivePrint() 
{
   printErrorMessage();
} 
