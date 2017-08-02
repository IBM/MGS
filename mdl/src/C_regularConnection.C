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

#include "C_regularConnection.h"
#include "C_connection.h"
#include "C_general.h"
#include "C_generalList.h"
#include "MdlContext.h"
#include "NotFoundException.h"
#include "PredicateException.h"
#include "Predicate.h"
#include "C_interfacePointerList.h"
#include "ConnectionException.h"
#include "ConnectionCCBase.h"
#include "StructType.h"
#include "Connection.h"
#include "RegularConnection.h"
#include "UserFunctionCall.h"
#include "SharedCCBase.h"
#include <memory>
#include <string>
#include <iostream>
#include <sstream>

void C_regularConnection::execute(MdlContext* context, 
				  ConnectionCCBase* connectionBase) 
{

   RegularConnection* regularConnection;

   regularConnection = new RegularConnection(
      _componentType, _directionType);
   
   doConnectionWork(context, connectionBase, regularConnection);

   if (_predicate) { 
      StructType* structType = 0; // inAttr or outAttr
      if (_directionType == Connection::_PRE) {
	 structType = connectionBase->getInAttrPSet();
      } else { // _POST
	 structType = connectionBase->getOutAttrPSet();
      }
      _predicate->setPSet(*structType);
      _predicate->setInstances(connectionBase->getInstances());

      SharedCCBase* shared = 0;
      shared = dynamic_cast<SharedCCBase*>(connectionBase);
      if (shared == 0) {
	 if(_predicate->checkShareds()) {
	    throw ConnectionException(
	       "in " + getTypeStr() + 
	       ", a shared member is being used in predicate.");
	 }
      } else {
	 _predicate->setShareds(shared->getShareds());
      }

      _predicate->setFunctionPredicateName(
	 connectionBase->getPredicateFunctions());

      try {
	 _predicate->getResult();
      } catch (PredicateException& e) {
	 std::ostringstream os;
	 os << "in " << getTypeStr() << ", " << e.getError();
	 throw ConnectionException(os.str());
      }
      std::auto_ptr<Predicate> predicate;
      predicate.reset(_predicate);
      _predicate = 0;
      regularConnection->setPredicate(predicate);
   } 

   if (_generalList->getUserFunctionCalls()) { 
      std::auto_ptr<std::vector<UserFunctionCall*> > userFunctionCalls;
      _generalList->releaseUserFunctionCalls(userFunctionCalls);
      std::vector<UserFunctionCall*>::const_iterator it, 
	 end = userFunctionCalls->end();
      for (it = userFunctionCalls->begin(); it != end; ++it) {
	 if (!connectionBase->userFunctionCallExists((*it)->getName())) {
	    std::ostringstream os;
	    os << "in " << getTypeStr() << ", " << (*it)->getName() 
	       << " is not defined as a UserFunction.";
	    throw ConnectionException(os.str());	    
	 }
      }
      regularConnection->setUserFunctionCalls(userFunctionCalls);
   }
   std::auto_ptr<RegularConnection> aConnection;
   aConnection.reset(regularConnection);
   
   connectionBase->addConnection(aConnection);
}


C_regularConnection::C_regularConnection() 
   : C_connection(), _predicate(0),
     _componentType(Connection::_EDGE), 
     _directionType(Connection::_PRE)    
{
}

C_regularConnection::C_regularConnection(
   C_interfacePointerList* ipl, C_generalList* gl,
   Connection::ComponentType componentType, 
   Connection::DirectionType directionType, Predicate* predicate) 
   : C_connection(ipl, gl), _predicate(predicate),
     _componentType(componentType), _directionType(directionType) 
{
}

C_regularConnection::C_regularConnection(const C_regularConnection& rv) 
   : C_connection(rv), _predicate(0),
     _componentType(rv._componentType), _directionType(rv._directionType)  
{
   if (rv._predicate) {
      std::auto_ptr<Predicate> dup;
      rv._predicate->duplicate(dup);
      _predicate = dup.release();
   }
}

void C_regularConnection::duplicate(
   std::auto_ptr<C_regularConnection>& rv) const
{
   rv.reset(new C_regularConnection(*this));
}

void C_regularConnection::duplicate(std::auto_ptr<C_connection>& rv) const
{
   rv.reset(new C_regularConnection(*this));
}

void C_regularConnection::duplicate(std::auto_ptr<C_general>& rv) const
{
   rv.reset(new C_regularConnection(*this));
}

C_regularConnection::~C_regularConnection() 
{
   delete _predicate;
}

void C_regularConnection::addToList(C_generalList* gl) 
{
   std::auto_ptr<C_regularConnection> con;
   con.reset(new C_regularConnection(*this));
   gl->addConnection(con);
}

std::string C_regularConnection::getTypeStr() const
{
   return "Connection";
}
