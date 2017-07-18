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

#include "C_interfaceToShared.h"
#include "C_interfaceMapping.h"
#include "C_generalList.h"
#include "MdlContext.h"
#include "InternalException.h"
#include <memory>
#include <string>

void C_interfaceToShared::execute(MdlContext* context) 
{
   C_interfaceMapping::execute(context);
}

void C_interfaceToShared::addToList(C_generalList* gl) 
{  
   std::auto_ptr<C_interfaceToShared> im;
   im.reset(new C_interfaceToShared(*this));
   gl->addInterfaceToShared(im);
}

C_interfaceToShared::C_interfaceToShared() 
   : C_interfaceMapping()
{
}

C_interfaceToShared::C_interfaceToShared(
   const std::string& interface, const std::string& interfaceMember,
   C_identifierList* member)
   : C_interfaceMapping(interface, interfaceMember, member)
{
} 

void C_interfaceToShared::duplicate(
   std::auto_ptr<C_interfaceToShared>& rv) const
{
   rv.reset(new C_interfaceToShared(*this));
}

void C_interfaceToShared::duplicate(
   std::auto_ptr<C_interfaceMapping>& rv) const
{
   rv.reset(new C_interfaceToShared(*this));
}

void C_interfaceToShared::duplicate(std::auto_ptr<C_general>& rv) const
{
   rv.reset(new C_interfaceToShared(*this));
}

C_interfaceToShared::~C_interfaceToShared() 
{
}


