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
      std::auto_ptr<TriggeredFunction> phase(
	 new TriggeredFunctionInstance(*it, _runType));
      gl->addTriggeredFunction(phase);
   }
}

C_triggeredFunctionInstance::C_triggeredFunctionInstance(
   C_identifierList* identifierList, TriggeredFunction::RunType runType) 
   : C_triggeredFunction(identifierList, runType)
{
} 

void C_triggeredFunctionInstance::duplicate(
   std::auto_ptr<C_triggeredFunction>& rv) const
{
   rv.reset(new C_triggeredFunctionInstance(*this));
}

void C_triggeredFunctionInstance::duplicate(std::auto_ptr<C_general>& rv) const
{
   rv.reset(new C_triggeredFunctionInstance(*this));
}

C_triggeredFunctionInstance::~C_triggeredFunctionInstance() 
{
}


