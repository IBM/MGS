// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "C_triggeredFunction.h"
#include "C_general.h"
#include "C_generalList.h"
#include "MdlContext.h"
#include "InternalException.h"
#include <memory>
#include <string>
#include "C_identifierList.h"

void C_triggeredFunction::execute(MdlContext* context) 
{
   if (_identifierList == 0) {
      throw InternalException(
	 "_identifierList is 0 in C_triggeredFunction::execute");
   }
   _identifierList->execute(context);
}

C_triggeredFunction::C_triggeredFunction(const C_triggeredFunction& rv)
   : _identifierList(0), _runType(rv._runType)
{
   copyOwnedHeap(rv);
}

C_triggeredFunction& C_triggeredFunction::operator=(const C_triggeredFunction& rv)
{
   if (this != &rv) {
      C_general::operator=(rv);
      destructOwnedHeap();
      copyOwnedHeap(rv);
      _runType = rv._runType;
   }
   return *this;
}

C_triggeredFunction::C_triggeredFunction(C_identifierList* identifierList, 
					 TriggeredFunction::RunType runType) 
   : C_general(), _identifierList(identifierList) , _runType(runType)
{
} 

C_triggeredFunction::~C_triggeredFunction() 
{
   destructOwnedHeap();
}

void C_triggeredFunction::copyOwnedHeap(const C_triggeredFunction& rv)
{
   if (rv._identifierList) {
      std::unique_ptr<C_identifierList> dup;
      rv._identifierList->duplicate(std::move(dup));
      _identifierList = dup.release();
   } else {
      _identifierList = 0;
   }
}

void C_triggeredFunction::destructOwnedHeap()
{
   delete _identifierList;
}
