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

#include "C_constant.h"
#include "C_interfaceImplementorBase.h"
#include "MdlContext.h"
#include "Constant.h"
#include "Generatable.h"
#include <memory>

void C_constant::execute(MdlContext* context) 
{
   Constant* cc = new Constant(getFileName());
   executeInterfaceImplementorBase(context, cc);
   cc->checkAllMemberToInterfaces();	 
   std::auto_ptr<Generatable> conMember;
   conMember.reset(cc);
   context->_generatables->addMember(_name, conMember);
}

C_constant::C_constant() 
   : C_interfaceImplementorBase() 
{

}

C_constant::C_constant(const std::string& name, C_interfacePointerList* ipl,
		       C_generalList* gl) 
   : C_interfaceImplementorBase(name, ipl, gl) 
{
}

C_constant::C_constant(const C_constant& rv) 
   : C_interfaceImplementorBase(rv) 
{
}

void C_constant::duplicate(std::auto_ptr<C_constant>& rv) const
{
   rv.reset(new C_constant(*this));
}

C_constant::~C_constant() 
{
}


