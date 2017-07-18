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

#include "RegularConnection.h"
#include "Connection.h"
#include "InternalException.h"
#include "Predicate.h"
#include "StructType.h"
#include "InterfaceToMember.h"
#include "Constants.h"
#include "UserFunctionCall.h"
#include <memory>
#include <string>
#include <sstream>
#include <vector>
#include <cassert>

RegularConnection::RegularConnection() 
   : Connection(_PRE, _EDGE), _predicate(0), _userFunctionCalls(0)
{
}

RegularConnection::RegularConnection(ComponentType componentType,
				     DirectionType directionType)
   : Connection(directionType, componentType), _predicate(0), 
     _userFunctionCalls(0)
{
}

RegularConnection::RegularConnection(const RegularConnection& rv) 
   : Connection(rv), _predicate(0), _userFunctionCalls(0)
{
}

RegularConnection& RegularConnection::operator=(const RegularConnection& rv)
{
   if (this != &rv) {
     Connection::operator=(rv);
      destructOwnedHeap();
      copyOwnedHeap(rv);
   }
   return *this;
}

void RegularConnection::duplicate(std::auto_ptr<RegularConnection>& rv) const
{
   rv.reset(new RegularConnection(*this));
}

void RegularConnection::duplicate(std::auto_ptr<Connection>& rv) const
{
   rv.reset(new RegularConnection(*this));
}

void RegularConnection::setPredicate(std::auto_ptr<Predicate>& pre) 
{
   delete _predicate;
   _predicate = pre.release();
}

void RegularConnection::setUserFunctionCalls(
   std::auto_ptr<std::vector<UserFunctionCall*> > userFunctionCall) 
{
   delete _userFunctionCalls;
   _userFunctionCalls = userFunctionCall.release();
}

RegularConnection::~RegularConnection() 
{
   destructOwnedHeap();
}

void RegularConnection::copyOwnedHeap(const RegularConnection& rv)
{
   if (rv._predicate) {
      std::auto_ptr<Predicate> dup;
      rv._predicate->duplicate(dup);
      _predicate = dup.release();
   } else {
      _predicate = 0;
   }

   if (rv._userFunctionCalls) {
      _userFunctionCalls = new std::vector<UserFunctionCall*>();
      std::vector<UserFunctionCall*>::const_iterator it
	 , end = rv._userFunctionCalls->end();
      std::auto_ptr<UserFunctionCall> dup;   
      for (it = rv._userFunctionCalls->begin(); it != end; ++it) {
	 (*it)->duplicate(dup);
	 _userFunctionCalls->push_back(dup.release());
      }
   } else {
      _userFunctionCalls = 0;
   }
}

void RegularConnection::destructOwnedHeap()
{
   delete _predicate;
   if (_userFunctionCalls) {
      std::vector<UserFunctionCall*>::iterator it, 
	 end = _userFunctionCalls->end();
      for (it = _userFunctionCalls->begin(); it != end; ++it) {
	 delete *it;
      }
      delete _userFunctionCalls;
      _userFunctionCalls = 0;
   }
}

std::string RegularConnection::getPredicateString() const
{
   std::ostringstream os;
   if (_predicate) {
      os << _predicate->getName(); 
   }
   return os.str();
}

std::string RegularConnection::getUserFunctionCallsString() const
{
   std::ostringstream os;
   if (_userFunctionCalls) {
      std::vector<UserFunctionCall*>::const_iterator it, 
	 end = _userFunctionCalls->end();
      for (it = _userFunctionCalls->begin(); it != end; ++it) {
	 os << "\t\t" << (*it)->getName() << "();\n";
      }
   }  
   return os.str();
}

std::string RegularConnection::getConnectionCode(
   const std::string& name, const std::string& functionParameters) const
{
   std::ostringstream os;
   std::string psetName;
   if (_directionType == _PRE) {
      psetName += INATTRPSETNAME;
   } else {
      psetName += OUTATTRPSETNAME;
   }
   std::string tab;   
   std::string predicateString;   
   if (_predicate) {
      tab = TAB + TAB;
      os << TAB << "if (";
      if (_predicate) {
       	 os << _predicate->getName();
				 predicateString = (_predicate->getName());
				 //predicateString = (_predicate->getPredicate1Name());
      }
      os << ") {\n";
//      os << getCommonConnectionCodeAlternativeInterfaceSet(tab, name);
   } else {
      tab = TAB;
//      os << getCommonConnectionCode(tab, name);
   }
	 //os << getCommonConnectionCodeAlternativeInterfaceSet(tab, name);
	 os << getCommonConnectionCodeAlternativeInterfaceSet(tab, name, predicateString);
	 if (_userFunctionCalls) {
     os << tab <<  "if (castMatchLocal) { \n";
     std::vector<UserFunctionCall*>::const_iterator it, 
       end = _userFunctionCalls->end();
     for (it = _userFunctionCalls->begin();  it != end; ++it) {
       //os << tab << (*it)->getName() << "(" << functionParameters << ");\n";
       os << tab << TAB << (*it)->getName() << "(" << functionParameters << ");\n";
     }
     os << tab << "}; \n";
   }
   if (_predicate) {
      os << TAB << "}\n";      
   }
   return os.str();
}

void RegularConnection::getFunctionPredicateNames(
   std::set<std::string>& names)const
{
   if (_predicate) {
      _predicate->getFunctionPredicateNames(names);
   }
}
