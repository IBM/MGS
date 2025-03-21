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

#include "C_interfaceToInstance.h"
#include "C_interfaceMapping.h"
#include "C_generalList.h"
#include "MdlContext.h"
#include "InternalException.h"
#include <memory>
#include <string>

void C_interfaceToInstance::execute(MdlContext* context) 
{
   C_interfaceMapping::execute(context);
}

void C_interfaceToInstance::addToList(C_generalList* gl) 
{  
   std::unique_ptr<C_interfaceToInstance> im;
   im.reset(new C_interfaceToInstance(*this));
   gl->addInterfaceToInstance(std::move(im));
}

C_interfaceToInstance::C_interfaceToInstance() 
   : C_interfaceMapping()
{
}

C_interfaceToInstance::C_interfaceToInstance(
   const std::string& interface, const std::string& interfaceMember,
   C_identifierList* member)
   : C_interfaceMapping(interface, interfaceMember, member)
{
} 

void C_interfaceToInstance::duplicate(
   std::unique_ptr<C_interfaceToInstance>&& rv) const
{
   rv.reset(new C_interfaceToInstance(*this));
}

void C_interfaceToInstance::duplicate(
   std::unique_ptr<C_interfaceMapping>&& rv) const
{
   rv.reset(new C_interfaceToInstance(*this));
}

void C_interfaceToInstance::duplicate(std::unique_ptr<C_general>&& rv) const
{
   rv.reset(new C_interfaceToInstance(*this));
}

C_interfaceToInstance::~C_interfaceToInstance() 
{
}


