// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
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
		 std::unique_ptr<PhaseType>&& phaseType) 
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
      std::unique_ptr<C_phaseIdentifierList> dup;
      rv._phaseIdentifierList->duplicate(std::move(dup));
      _phaseIdentifierList = dup.release();
   } else {
      _phaseIdentifierList = 0;
   }
   if (rv._phaseType) {
      std::unique_ptr<PhaseType> dup;
      rv._phaseType->duplicate(std::move(dup));
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
