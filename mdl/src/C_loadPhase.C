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

#include "C_loadPhase.h"
#include "C_phase.h"
#include "C_generalList.h"
#include "MdlContext.h"
#include "LoadPhase.h"
#include "PhaseType.h"
#include "C_phaseIdentifierList.h"
#include <memory>
#include <string>

void C_loadPhase::execute(MdlContext* context) 
{
   C_phase::execute(context);
}

void C_loadPhase::addToList(C_generalList* gl) 
{
   std::unique_ptr<PhaseType> dup;
   const std::vector<C_phaseIdentifier*>& 
      ids = _phaseIdentifierList->getPhaseIdentifiers();

   std::vector<C_phaseIdentifier*>::const_iterator it, end = ids.end();
   for (it = ids.begin(); it != end; ++it) {   
      _phaseType->duplicate(std::move(dup));
      std::unique_ptr<Phase> phase = std::make_unique<LoadPhase>(
         (*it)->getName(), std::move(dup), (*it)->getIdentifiers());
      gl->addPhase(std::move(phase));
   }
}

C_loadPhase::C_loadPhase(C_phaseIdentifierList* phaseIdentifierList, 
			 std::unique_ptr<PhaseType>&& phaseType) 
   : C_phase(phaseIdentifierList, std::move(phaseType))
{
} 

void C_loadPhase::duplicate(std::unique_ptr<C_phase>&& rv) const
{
   rv.reset(new C_loadPhase(*this));
}

void C_loadPhase::duplicate(std::unique_ptr<C_general>&& rv) const
{
   rv.reset(new C_loadPhase(*this));
}

C_loadPhase::~C_loadPhase() 
{
}


