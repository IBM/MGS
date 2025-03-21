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

#include "C_userFunction.h"
#include "C_general.h"
#include "C_generalList.h"
#include "MdlContext.h"
#include "InternalException.h"
#include "UserFunction.h"
#include "C_identifierList.h"

#include <memory>
#include <string>

void C_userFunction::execute(MdlContext* context) 
{
   if (_identifierList == 0) {
      throw InternalException(
	 "_identifierList is 0 in C_userFunction::execute");
   }
   _identifierList->execute(context);
}

void C_userFunction::addToList(C_generalList* gl) 
{
   const std::vector<std::string>& ids = _identifierList->getIdentifiers();
   std::vector<std::string>::const_iterator it, end = ids.end();
   for (it = ids.begin(); it != end; ++ it) {
      std::unique_ptr<UserFunction> userFunction(
	 new UserFunction(*it));
      gl->addUserFunction(std::move(userFunction));
   }
}


C_userFunction::C_userFunction() 
   : C_general(), _identifierList(0) 
{

}

C_userFunction::C_userFunction(C_identifierList* identifierList) 
   : C_general(), _identifierList(identifierList) 
{

} 

C_userFunction::C_userFunction(const C_userFunction& rv) 
   : C_general(rv)
{
   copyOwnedHeap(rv);
}

C_userFunction& C_userFunction::operator=(const C_userFunction& rv)
{
   if (this != &rv) {
      C_general::operator=(rv);
      destructOwnedHeap();
      copyOwnedHeap(rv);
   }
   return *this;
}

void C_userFunction::duplicate(std::unique_ptr<C_userFunction>&& rv) const
{
   rv.reset(new C_userFunction(*this));
}

void C_userFunction::duplicate(std::unique_ptr<C_general>&& rv) const
{
   rv.reset(new C_userFunction(*this));
}

C_userFunction::~C_userFunction() 
{
   destructOwnedHeap();
}

void C_userFunction::copyOwnedHeap(const C_userFunction& rv)
{
   if (rv._identifierList) {
      std::unique_ptr<C_identifierList> dup;
      rv._identifierList->duplicate(std::move(dup));
      _identifierList = dup.release();
   } else {
      _identifierList = 0;
   }
}

void C_userFunction::destructOwnedHeap()
{
   delete _identifierList;
}

