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

#include "C_phase_mapping.h"
#include "SyntaxError.h"
#include "C_production.h"
#include "PhaseDataItem.h"
#include "Phase.h"
#include "LensContext.h"
#include "SyntaxErrorException.h"
#include "CompCategoryBase.h"
#include "PhaseDataItem.h"
#include "Phase.h"

#include <iostream>

void C_phase_mapping::internalExecute(LensContext *c)
{
   CompCategoryBase* ccBase = c->getCurrentCompCategoryBase();
   std::string type = ccBase->getPhaseType(_modelPhase);
   if (type == "") {
      throwError(_modelPhase + " does not exist in CompCategory.==================");
   }
   DataItem* dataItem = dynamic_cast<DataItem*> ( 
      c->symTable.getEntry(_simulationPhase));
   if (dataItem == 0) {
      throwError(_simulationPhase + " is not defined.++++++++++++++");
   }
   PhaseDataItem* phaseDataItem = dynamic_cast<PhaseDataItem*> (dataItem);
   if (phaseDataItem == 0) {
      throwError(_simulationPhase + " is not defined as phase.");
   }
   if (type != phaseDataItem->getPhase()->getType()) {
      throwError(_modelPhase + " type: " + type + ", " + _simulationPhase +
		 " type: " + phaseDataItem->getPhase()->getType() + ".");
   }
   std::auto_ptr<Phase> dup;
   phaseDataItem->getPhase()->duplicate(dup);
   ccBase->addPhaseMapping(_modelPhase, dup);
}

C_phase_mapping::C_phase_mapping(const C_phase_mapping& rv)
   : C_production(rv), _modelPhase(rv._modelPhase), 
     _simulationPhase(rv._simulationPhase)
{
}

C_phase_mapping::C_phase_mapping(const std::string& modelPhase, 
				 const std::string& simulationPhase, 
				 SyntaxError * error)
   : C_production(error), _modelPhase(modelPhase), 
     _simulationPhase(simulationPhase)
{
}


C_phase_mapping* C_phase_mapping::duplicate() const
{
   return new C_phase_mapping(*this);
}


C_phase_mapping::~C_phase_mapping()
{
}

void C_phase_mapping::checkChildren() 
{
} 

void C_phase_mapping::recursivePrint() 
{
   printErrorMessage();
} 
