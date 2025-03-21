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

#include "C_connectionCCBase.h"
#include "C_compCategoryBase.h"
#include "C_generalList.h"
#include "C_regularConnection.h"
#include "CompCategoryBase.h"
#include "ConnectionException.h"
#include "SyntaxErrorException.h"
#include "C_interfacePointerList.h"
#include "MdlContext.h"
#include "ConnectionCCBase.h"
#include "Interface.h"
#include <memory>
#include <vector>
#include <string>
#include <iostream>

void C_connectionCCBase::execute(MdlContext* context) 
{
   // look at: void C_connectionCCBase::
   // executeConnectionCCBase(MdlContext* context, ConnectionCCBase* cc) 
}

C_connectionCCBase::C_connectionCCBase() : C_compCategoryBase() 
{

}

C_connectionCCBase::C_connectionCCBase(const std::string& name, 
				       C_interfacePointerList* ipl
			 , C_generalList* gl) 
   : C_compCategoryBase(name, ipl, gl) 
{

}


C_connectionCCBase::C_connectionCCBase(const C_connectionCCBase& rv) 
   : C_compCategoryBase(rv)  
{
}

void C_connectionCCBase::duplicate(std::unique_ptr<C_compCategoryBase>&& rv) const
{
   rv.reset(new C_connectionCCBase(*this));
}

void C_connectionCCBase::duplicate(std::unique_ptr<C_connectionCCBase>&& rv) const
{
   rv.reset(new C_connectionCCBase(*this));
}

void C_connectionCCBase::executeConnectionCCBase(MdlContext* context, 
						 ConnectionCCBase* cc) const
{
   if (_generalList->getUserFunctions()) { 
      std::unique_ptr<std::vector<UserFunction*> > userFunctions;
      _generalList->releaseUserFunctions(userFunctions);
      cc->setUserFunctions(userFunctions);
   }
   if (_generalList->getPredicateFunctions()) { 
      std::unique_ptr<std::vector<PredicateFunction*> > predicateFunctions;
      _generalList->releasePredicateFunctions(predicateFunctions);
      cc->setPredicateFunctions(predicateFunctions);
   }
   if (_generalList->getConnectionVec()) {
      std::unique_ptr<std::vector<C_regularConnection*> > connectionVec;
      _generalList->releaseConnectionVec(connectionVec);
      std::vector<C_regularConnection*>::iterator it, 
	 end = connectionVec->end();
      for (it = connectionVec->begin(); it != end; it++) {
	 try {
	    (*it)->execute(context, cc);
	 } catch (ConnectionException &e) {
	    std::string mes = "In " + cc->getName() + " " 
	       + e.getError() + "\n";
	    //e.setError(mes);
	    throw SyntaxErrorException(mes);
	 } 
	 delete *it;
      }
   }
}


C_connectionCCBase::~C_connectionCCBase() 
{
}


