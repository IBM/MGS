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
      std::auto_ptr<TriggeredFunction> phase(
	 new TriggeredFunctionShared(*it, _runType));
      gl->addTriggeredFunction(phase);
   }
}

C_triggeredFunctionShared::C_triggeredFunctionShared(
   C_identifierList* identifierList, TriggeredFunction::RunType runType) 
   : C_triggeredFunction(identifierList, runType)
{
} 

void C_triggeredFunctionShared::duplicate(
   std::auto_ptr<C_triggeredFunction>& rv) const
{
   rv.reset(new C_triggeredFunctionShared(*this));
}

void C_triggeredFunctionShared::duplicate(std::auto_ptr<C_general>& rv) const
{
   rv.reset(new C_triggeredFunctionShared(*this));
}

C_triggeredFunctionShared::~C_triggeredFunctionShared() 
{
}


