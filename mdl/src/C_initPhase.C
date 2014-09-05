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

#include "C_initPhase.h"
#include "C_phase.h"
#include "C_generalList.h"
#include "MdlContext.h"
#include "InitPhase.h"
#include "PhaseType.h"
#include "C_phaseIdentifierList.h"
#include <memory>
#include <string>

void C_initPhase::execute(MdlContext* context) 
{
   C_phase::execute(context);
}

void C_initPhase::addToList(C_generalList* gl) 
{
   std::auto_ptr<PhaseType> dup;
   const std::vector<C_phaseIdentifier*>& 
      ids = _phaseIdentifierList->getPhaseIdentifiers();

   std::vector<C_phaseIdentifier*>::const_iterator it, end = ids.end();
   for (it = ids.begin(); it != end; ++it) {   
      _phaseType->duplicate(dup);
      std::auto_ptr<Phase> phase(
	 new InitPhase((*it)->getName(), dup, (*it)->getIdentifiers()));
      gl->addPhase(phase);
   }
}

C_initPhase::C_initPhase(C_phaseIdentifierList* phaseIdentifierList, 
			 std::auto_ptr<PhaseType>& phaseType) 
   : C_phase(phaseIdentifierList, phaseType)
{
} 

void C_initPhase::duplicate(std::auto_ptr<C_phase>& rv) const
{
   rv.reset(new C_initPhase(*this));
}

void C_initPhase::duplicate(std::auto_ptr<C_general>& rv) const
{
   rv.reset(new C_initPhase(*this));
}

C_initPhase::~C_initPhase() 
{
}


