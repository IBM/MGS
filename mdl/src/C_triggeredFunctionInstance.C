// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "C_triggeredFunctionInstance.h"
#include "C_generalList.h"
#include "MdlContext.h"
#include "TriggeredFunctionInstance.h"
#include "C_identifierList.h"
#include <memory>
#include <string>

void C_triggeredFunctionInstance::execute(MdlContext* context) 
{
   C_triggeredFunction::execute(context);
}

void C_triggeredFunctionInstance::addToList(C_generalList* gl) 
{
   const std::vector<std::string>& ids = _identifierList->getIdentifiers();
   std::vector<std::string>::const_iterator it, end = ids.end();
   for (it = ids.begin(); it != end; ++it) {   
      std::unique_ptr<TriggeredFunction> phase(
	 new TriggeredFunctionInstance(*it, _runType));
      gl->addTriggeredFunction(std::move(phase));
   }
}

C_triggeredFunctionInstance::C_triggeredFunctionInstance(
   C_identifierList* identifierList, TriggeredFunction::RunType runType) 
   : C_triggeredFunction(identifierList, runType)
{
} 

void C_triggeredFunctionInstance::duplicate(
   std::unique_ptr<C_triggeredFunction>&& rv) const
{
   rv.reset(new C_triggeredFunctionInstance(*this));
}

void C_triggeredFunctionInstance::duplicate(std::unique_ptr<C_general>&& rv) const
{
   rv.reset(new C_triggeredFunctionInstance(*this));
}

C_triggeredFunctionInstance::~C_triggeredFunctionInstance() 
{
}


