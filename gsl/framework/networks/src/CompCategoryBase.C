// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifdef HAVE_MPI
#include <mpi.h>
#endif
#include "CompCategoryBase.h"
//#include "ParameterSet.h"
#include "Simulation.h"
#include "WorkUnit.h"
#include "SyntaxErrorException.h"
#include "PhaseDataItem.h"
#include "Phase.h"
#include "LensContext.h"

#include <cassert>
#include <vector>
#include <cassert>

#ifdef AIX
#include <iostream>
#endif


CompCategoryBase::CompCategoryBase(
   Simulation& sim)
   : CompCategory(), _sim(sim)
{
   _sim.registerCompCat(this);
}

CompCategoryBase::~CompCategoryBase()
{
   std::map<std::string, std::deque<WorkUnit*> >::iterator mapIt, 
      mapEnd = _workUnits.end();
   for (mapIt = _workUnits.begin(); mapIt != mapEnd; ++mapIt) {
      std::deque<WorkUnit*>::iterator it, end = mapIt->second.end();
      for (it = mapIt->second.begin(); it != end; ++it) {
	 delete *it;
      }
   }

   std::map<std::string, Phase*>::iterator pIt, pEnd = _phaseMappings.end();
   for (pIt = _phaseMappings.begin(); pIt != pEnd; ++pIt) {
      delete pIt->second;
   }
}

void CompCategoryBase::addPhaseMapping(const std::string& name, 
				       std::unique_ptr<Phase>& phase)
{
   std::map<std::string, Phase*>::iterator it = _phaseMappings.find(name);
   if (it != _phaseMappings.end()) { 
      _phaseMappings[name] = phase.release();
   } else {
      std::string error = name + " does not exist in CompCategory.";
      throw SyntaxErrorException(error);
   }
}

std::string CompCategoryBase::getSimulationPhaseName(const std::string& name) {
   std::string retVal = "";
   if (!_phaseMappings.empty()) {
   std::map<std::string, Phase*>::iterator it = _phaseMappings.find(name);   
   if (it != _phaseMappings.end()) { 
      retVal = _phaseMappings[name]->getName();
   } else {
      std::string error = name + " does not exist in CompCategory.";
      throw SyntaxErrorException(error);
   }
   }
   return retVal;  
}

void CompCategoryBase::initializePhase(const std::string& name, 
				       const std::string& type,
				       bool communicating)
{
   _phaseMappings[name] = 0;
   _phaseTypes[name] = type;
   _phaseCommunicationTable[name] = communicating;
}

std::string CompCategoryBase::getPhaseType(const std::string& name)
{
   std::map<std::string, std::string>::iterator it = _phaseTypes.find(name);
   if (it != _phaseTypes.end()) {
      return it->second;
   }
   return "";
}

void CompCategoryBase::setUnmappedPhases(LensContext* c)
{
   std::map<std::string, Phase*>::iterator it, 
      end = _phaseMappings.end();
   for (it = _phaseMappings.begin(); it != end; ++it) {
      if (it->second == 0) {
	 DataItem* dataItem = dynamic_cast<DataItem*> ( 
	    c->symTable.getEntry(it->first));
	 if (dataItem == 0) {
	    std::string mes = it->first + " is not defined.";
	    throw SyntaxErrorException(mes);
	 }
	 PhaseDataItem* phaseDataItem = 
	    dynamic_cast<PhaseDataItem*> (dataItem);
	 if (phaseDataItem == 0) {
	    std::string mes = it->first + " is not defined as phase.";
	    throw SyntaxErrorException(mes);
	 }
	 std::string type = _phaseTypes[it->first];
	 if (type != phaseDataItem->getPhase()->getType()) {
	    std::string mes = 
	       "CompCategory " + it->first + " type: " + type + ", " + 
	       "Simulation " +  it->first + " type: " + 
	       phaseDataItem->getPhase()->getType() + ".";
	    throw SyntaxErrorException(mes);
	 } 
	 std::unique_ptr<Phase> dup;
	 phaseDataItem->getPhase()->duplicate(dup);
	 addPhaseMapping(it->first, dup);	 
      }
   }
}
