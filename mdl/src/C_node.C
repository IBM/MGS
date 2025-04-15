// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "C_node.h"
#include "C_sharedCCBase.h"
#include "C_generalList.h"
#include "MdlContext.h"
#include "Node.h"
#include "Generatable.h"
#include "ConnectionIncrement.h"
#include <memory>

void C_node::execute(MdlContext* context) 
{
   Node* cc = new Node(getFileName());
   executeInterfaceImplementorBase(context, cc);
   executeCompCategoryBase(context, cc);
   executeSharedCCBase(context, cc);
   executeConnectionCCBase(context, cc);
   cc->checkAllMemberToInterfaces();	 
   
   std::unique_ptr<Generatable> nodeMember;
   nodeMember.reset(cc);
   context->_generatables->addMember(_name, nodeMember);
}

C_node::C_node() 
   : C_sharedCCBase() 
{

}

C_node::C_node(const std::string& name, C_interfacePointerList* ipl
		       , C_generalList* gl) : C_sharedCCBase(name, ipl, gl) 
{

}

C_node::C_node(const C_node& rv) 
   : C_sharedCCBase(rv) 
{

}

void C_node::duplicate(std::unique_ptr<C_node>&& rv) const
{
   rv.reset(new C_node(*this));
}

C_node::~C_node() 
{

}
