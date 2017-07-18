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

#include "C_psetToInstance.h"
#include "C_psetMapping.h"
#include "C_generalList.h"
#include "MdlContext.h"
#include "InternalException.h"
#include <memory>
#include <string>

void C_psetToInstance::execute(MdlContext* context) 
{
   C_psetMapping::execute(context);
}

void C_psetToInstance::addToList(C_generalList* gl) 
{  
   std::auto_ptr<C_psetToInstance> im;
   im.reset(new C_psetToInstance(*this));
   gl->addPSetToInstance(im);
}

C_psetToInstance::C_psetToInstance() 
   : C_psetMapping()
{
}

C_psetToInstance::C_psetToInstance(const std::string& psetMember,
				   C_identifierList* member)
   : C_psetMapping(psetMember, member)
{
} 

void C_psetToInstance::duplicate(
   std::auto_ptr<C_psetToInstance>& rv) const
{
   rv.reset(new C_psetToInstance(*this));
}

void C_psetToInstance::duplicate(
   std::auto_ptr<C_psetMapping>& rv) const
{
   rv.reset(new C_psetToInstance(*this));
}

void C_psetToInstance::duplicate(std::auto_ptr<C_general>& rv) const
{
   rv.reset(new C_psetToInstance(*this));
}

C_psetToInstance::~C_psetToInstance() 
{
}


