// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BCM-YKT-11-19-2015
//
// (C) Copyright IBM Corp. 2005-2015  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#include "C_edge.h"
#include "C_sharedCCBase.h"
#include "C_generalList.h"
#include "C_edgeConnection.h"
#include "MdlContext.h"
#include "Edge.h"
#include "Generatable.h"
#include "ConnectionException.h"
#include "SyntaxErrorException.h"
#include <memory>
#include <iostream>

void C_edge::execute(MdlContext* context) 
{
   Edge* cc = new Edge(getFileName());
   executeInterfaceImplementorBase(context, cc);
   executeCompCategoryBase(context, cc);
   executeSharedCCBase(context, cc);
   executeConnectionCCBase(context, cc);
   if (_generalList->getPreNode()) {
      try {
	 _generalList->getPreNode()->execute(context, cc);
      } catch (ConnectionException &e) {
	 std::string mes = "In " + cc->getName() + " " + e.getError() + "\n";
	 //e.setError(mes);
	 throw SyntaxErrorException(mes);
      } 
   }
   if (_generalList->getPostNode()) {
      try {
	 _generalList->getPostNode()->execute(context, cc);
      } catch (ConnectionException &e) {
	 std::string mes = "In " + cc->getName() + " " + e.getError() + "\n";
	 //e.setError(mes);
	 throw SyntaxErrorException(mes);
      } 
   }
   cc->checkAllMemberToInterfaces();	 
   std::auto_ptr<Generatable> sharedMember;
   sharedMember.reset(cc);
   context->_generatables->addMember(_name, sharedMember);
}

C_edge::C_edge() : C_sharedCCBase() 
{

}

C_edge::C_edge(const std::string& name, C_interfacePointerList* ipl
	       , C_generalList* gl) 
   : C_sharedCCBase(name, ipl, gl) 
{

}

C_edge::C_edge(const C_edge& rv) 
   : C_sharedCCBase(rv) 
{

}

void C_edge::duplicate(std::auto_ptr<C_edge>& rv) const
{
   rv.reset(new C_edge(*this));
}

C_edge::~C_edge() 
{

}



