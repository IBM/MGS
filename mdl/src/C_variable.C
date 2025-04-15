// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "C_variable.h"
#include "C_connectionCCBase.h"
#include "MdlContext.h"
#include "Variable.h"
#include "Generatable.h"
#include <memory>

void C_variable::execute(MdlContext* context) 
{
   Variable* cc = new Variable(getFileName());
   executeInterfaceImplementorBase(context, cc);
   executeCompCategoryBase(context, cc);
   executeConnectionCCBase(context, cc);
   cc->checkAllMemberToInterfaces();	 
   std::unique_ptr<Generatable> conMember;
   conMember.reset(cc);
   context->_generatables->addMember(_name, conMember);
}

C_variable::C_variable() 
   : C_connectionCCBase() 
{

}

C_variable::C_variable(const std::string& name, C_interfacePointerList* ipl,
		       C_generalList* gl) 
   : C_connectionCCBase(name, ipl, gl) 
{

}

C_variable::C_variable(const C_variable& rv) 
   : C_connectionCCBase(rv) 
{

}

void C_variable::duplicate(std::unique_ptr<C_variable>&& rv) const
{
   rv.reset(new C_variable(*this));
}

C_variable::~C_variable() 
{

}
