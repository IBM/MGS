// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BCM-YKT-11-19-2015
//
// (C) Copyright IBM Corp. 2005-2015  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#include "C_finalPhase.h"
#include "C_phase.h"
#include "C_generalList.h"
#include "MdlContext.h"
#include "FinalPhase.h"
#include "PhaseType.h"
#include "C_phaseIdentifierList.h"
#include <memory>
#include <string>

void C_finalPhase::execute(MdlContext* context) 
{
   C_phase::execute(context);
}

void C_finalPhase::addToList(C_generalList* gl) 
{
   std::auto_ptr<PhaseType> dup;
   const std::vector<C_phaseIdentifier*>& 
      ids = _phaseIdentifierList->getPhaseIdentifiers();

   std::vector<C_phaseIdentifier*>::const_iterator it, end = ids.end();
   for (it = ids.begin(); it != end; ++it) {   
      _phaseType->duplicate(dup);
      std::auto_ptr<Phase> phase(
	 new FinalPhase((*it)->getName(), dup, (*it)->getIdentifiers()));
      gl->addPhase(phase);
   }
}

C_finalPhase::C_finalPhase(C_phaseIdentifierList* phaseIdentifierList, 
			   std::auto_ptr<PhaseType>& phaseType) 
   : C_phase(phaseIdentifierList, phaseType)
{
} 

void C_finalPhase::duplicate(std::auto_ptr<C_phase>& rv) const
{
   rv.reset(new C_finalPhase(*this));
}

void C_finalPhase::duplicate(std::auto_ptr<C_general>& rv) const
{
   rv.reset(new C_finalPhase(*this));
}

C_finalPhase::~C_finalPhase() 
{
}


