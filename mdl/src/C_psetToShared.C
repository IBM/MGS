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

#include "C_psetToShared.h"
#include "C_psetMapping.h"
#include "C_generalList.h"
#include "MdlContext.h"
#include "InternalException.h"
#include <memory>
#include <string>

void C_psetToShared::execute(MdlContext* context) 
{
   C_psetMapping::execute(context);
}

void C_psetToShared::addToList(C_generalList* gl) 
{  
   std::auto_ptr<C_psetToShared> im;
   im.reset(new C_psetToShared(*this));
   gl->addPSetToShared(im);
}

C_psetToShared::C_psetToShared() 
   : C_psetMapping()
{
}

C_psetToShared::C_psetToShared(const std::string& psetMember,  
			       C_identifierList* member)
   : C_psetMapping(psetMember, member)
{
} 

void C_psetToShared::duplicate(
   std::auto_ptr<C_psetToShared>& rv) const
{
   rv.reset(new C_psetToShared(*this));
}

void C_psetToShared::duplicate(
   std::auto_ptr<C_psetMapping>& rv) const
{
   rv.reset(new C_psetToShared(*this));
}

void C_psetToShared::duplicate(std::auto_ptr<C_general>& rv) const
{
   rv.reset(new C_psetToShared(*this));
}

C_psetToShared::~C_psetToShared() 
{
}


