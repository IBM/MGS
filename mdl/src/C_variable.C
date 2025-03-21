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
