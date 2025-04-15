// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "C_runtimePhase.h"
#include "C_phase.h"
#include "C_generalList.h"
#include "MdlContext.h"
#include "RuntimePhase.h"
#include "PhaseType.h"
#include "C_phaseIdentifierList.h"
#include <memory>
#include <string>

void C_runtimePhase::execute(MdlContext* context) 
{
   C_phase::execute(context);
}

void C_runtimePhase::addToList(C_generalList* gl) 
{
   std::unique_ptr<PhaseType> dup;
   const std::vector<C_phaseIdentifier*>& 
      ids = _phaseIdentifierList->getPhaseIdentifiers();

   std::vector<C_phaseIdentifier*>::const_iterator it, end = ids.end();
   for (it = ids.begin(); it != end; ++it) {   
      _phaseType->duplicate(std::move(dup));
      std::unique_ptr<Phase>phase = std::make_unique<RuntimePhase>(
         (*it)->getName(), std::move(dup), (*it)->getIdentifiers());
      gl->addPhase(std::move(phase));
   }
}

C_runtimePhase::C_runtimePhase(C_phaseIdentifierList* phaseIdentifierList, 
			       std::unique_ptr<PhaseType>&& phaseType) 
   : C_phase(phaseIdentifierList, std::move(phaseType))
{
} 

void C_runtimePhase::duplicate(std::unique_ptr<C_phase>&& rv) const
{
   rv.reset(new C_runtimePhase(*this));
}

void C_runtimePhase::duplicate(std::unique_ptr<C_general>&& rv) const
{
   rv.reset(new C_runtimePhase(*this));
}

C_runtimePhase::~C_runtimePhase() 
{
}


