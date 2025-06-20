// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "C_triggeredFunctionShared.h"
#include "C_generalList.h"
#include "MdlContext.h"
#include "TriggeredFunctionShared.h"
#include "C_identifierList.h"
#include <memory>
#include <string>

void C_triggeredFunctionShared::execute(MdlContext* context) 
{
   C_triggeredFunction::execute(context);
}

void C_triggeredFunctionShared::addToList(C_generalList* gl) 
{
   const std::vector<std::string>& ids = _identifierList->getIdentifiers();
   std::vector<std::string>::const_iterator it, end = ids.end();
   for (it = ids.begin(); it != end; ++it) {   
      std::unique_ptr<TriggeredFunction> phase(
	 new TriggeredFunctionShared(*it, _runType));
      gl->addTriggeredFunction(std::move(phase));
   }
}

C_triggeredFunctionShared::C_triggeredFunctionShared(
   C_identifierList* identifierList, TriggeredFunction::RunType runType) 
   : C_triggeredFunction(identifierList, runType)
{
} 

void C_triggeredFunctionShared::duplicate(
   std::unique_ptr<C_triggeredFunction>&& rv) const
{
   rv.reset(new C_triggeredFunctionShared(*this));
}

void C_triggeredFunctionShared::duplicate(std::unique_ptr<C_general>&& rv) const
{
   rv.reset(new C_triggeredFunctionShared(*this));
}

C_triggeredFunctionShared::~C_triggeredFunctionShared() 
{
}


