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
   std::auto_ptr<PhaseType> dup;
   const std::vector<C_phaseIdentifier*>& 
      ids = _phaseIdentifierList->getPhaseIdentifiers();

   std::vector<C_phaseIdentifier*>::const_iterator it, end = ids.end();
   for (it = ids.begin(); it != end; ++it) {   
      _phaseType->duplicate(dup);
      std::auto_ptr<Phase> phase(
	 new LoadPhase((*it)->getName(), dup, (*it)->getIdentifiers()));
      gl->addPhase(phase);
   }
}

C_loadPhase::C_loadPhase(C_phaseIdentifierList* phaseIdentifierList, 
			 std::auto_ptr<PhaseType>& phaseType) 
   : C_phase(phaseIdentifierList, phaseType)
{
} 

void C_loadPhase::duplicate(std::auto_ptr<C_phase>& rv) const
{
   rv.reset(new C_loadPhase(*this));
}

void C_loadPhase::duplicate(std::auto_ptr<C_general>& rv) const
{
   rv.reset(new C_loadPhase(*this));
}

C_loadPhase::~C_loadPhase() 
{
}


