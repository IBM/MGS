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
#include "C_general.h"
#include "C_generalList.h"
#include "MdlContext.h"
#include "InternalException.h"
#include "PhaseType.h"
#include <memory>
#include <string>
#include "C_phaseIdentifierList.h"

void C_phase::execute(MdlContext* context) 
{
   if (_phaseIdentifierList == 0) {
      throw InternalException(
	 "_phaseIdentifierList is 0 in C_phase::execute");
   }
   _phaseIdentifierList->execute(context);
}

C_phase::C_phase(const C_phase& rv)
   : _phaseIdentifierList(0), _phaseType(0)
{
   copyOwnedHeap(rv);
}

C_phase& C_phase::operator=(const C_phase& rv)
{
   if (this != &rv) {
      C_general::operator=(rv);
      destructOwnedHeap();
      copyOwnedHeap(rv);
   }
   return *this;
}

C_phase::C_phase(C_phaseIdentifierList* phaseIdentifierList, 
		 std::auto_ptr<PhaseType>& phaseType) 
   : C_general(), _phaseIdentifierList(phaseIdentifierList) 
{
   _phaseType = phaseType.release();
} 

C_phase::~C_phase() 
{
   destructOwnedHeap();
}

void C_phase::copyOwnedHeap(const C_phase& rv)
{
   if (rv._phaseIdentifierList) {
      std::auto_ptr<C_phaseIdentifierList> dup;
      rv._phaseIdentifierList->duplicate(dup);
      _phaseIdentifierList = dup.release();
   } else {
      _phaseIdentifierList = 0;
   }
   if (rv._phaseType) {
      std::auto_ptr<PhaseType> dup;
      rv._phaseType->duplicate(dup);
      _phaseType = dup.release();
   } else {
      _phaseType = 0;
   }
}

void C_phase::destructOwnedHeap()
{
   delete _phaseIdentifierList;
   delete _phaseType;
}
