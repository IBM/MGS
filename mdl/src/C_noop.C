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

#include "C_noop.h"
#include "C_general.h"
#include "C_generalList.h"
#include "MdlContext.h"
#include <memory>
#include <string>

void C_noop::execute(MdlContext* context) 
{
}

void C_noop::addToList(C_generalList* gl) 
{
}


C_noop::C_noop() : C_general() 
{

}

C_noop::C_noop(const C_noop& rv) 
   : C_general(rv) 
{
}

void C_noop::duplicate(std::auto_ptr<C_noop>& rv) const
{
   rv.reset(new C_noop(*this));
}

void C_noop::duplicate(std::auto_ptr<C_general>& rv) const
{
   rv.reset(new C_noop(*this));
}

C_noop::~C_noop() 
{
}


