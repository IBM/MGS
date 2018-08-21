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
   
   std::auto_ptr<Generatable> nodeMember;
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

void C_node::duplicate(std::auto_ptr<C_node>& rv) const
{
   rv.reset(new C_node(*this));
}

C_node::~C_node() 
{

}
