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

#include "C_sharedMapping.h"
#include "C_interfaceMapping.h"
#include "C_generalList.h"
#include "MdlContext.h"
#include "InternalException.h"
#include <memory>
#include <string>

void C_sharedMapping::execute(MdlContext* context) 
{
   C_interfaceMapping::execute(context);
}

void C_sharedMapping::addToList(C_generalList* gl) 
{  
   std::auto_ptr<C_sharedMapping> sm;
   sm.reset(new C_sharedMapping(*this));
   gl->addSharedMapping(sm);
}


C_sharedMapping::C_sharedMapping() 
   : C_interfaceMapping()
{
}

C_sharedMapping::C_sharedMapping(const std::string& interface,
				 const std::string& interfaceMember,
				 C_identifierList* dataType,
				 bool amp)
   : C_interfaceMapping(interface, interfaceMember, dataType, amp)
{
} 

void C_sharedMapping::duplicate(std::auto_ptr<C_sharedMapping>& rv) const
{
   rv.reset(new C_sharedMapping(*this));
}

void C_sharedMapping::duplicate(std::auto_ptr<C_interfaceMapping>& rv) const
{
   rv.reset(new C_sharedMapping(*this));
}

void C_sharedMapping::duplicate(std::auto_ptr<C_general>& rv) const
{
   rv.reset(new C_sharedMapping(*this));
}

C_sharedMapping::~C_sharedMapping() 
{
}


