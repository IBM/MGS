// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
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
   std::unique_ptr<Generatable> conMember;
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

void C_constant::duplicate(std::unique_ptr<C_constant>&& rv) const
{
   rv.reset(new C_constant(*this));
}

C_constant::~C_constant() 
{
}


