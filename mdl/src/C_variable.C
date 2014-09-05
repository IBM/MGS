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
   std::auto_ptr<Generatable> conMember;
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

void C_variable::duplicate(std::auto_ptr<C_variable>& rv) const
{
   rv.reset(new C_variable(*this));
}

C_variable::~C_variable() 
{

}
