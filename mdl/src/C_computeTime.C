// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "C_computeTime.h"
#include "C_general.h"
#include "C_generalList.h"
#include "MdlContext.h"
#include "InternalException.h"
#include "UserFunction.h"
#include "C_identifierList.h"

#include <memory>
#include <string>
#include <float.h>
#include <iostream>

void C_computeTime::execute(MdlContext* context) 
{
#if 0
   if (_identifierList == 0) {
      throw InternalException(
	 "_identifierList is 0 in C_computeTime::execute");
   }
   _identifierList->execute(context);
#endif
}

void C_computeTime::addToList(C_generalList* gl) 
{
   const std::vector<std::string>& ids = _identifierList->getIdentifiers();
   std::vector<std::string>::const_iterator it, end = ids.end();
   for (it = ids.begin(); it != end; ++ it) {
      std::unique_ptr<UserFunction> computeTime(
	 new UserFunction(*it));
      gl->addUserFunction(std::move(computeTime));
   }
}


C_computeTime::C_computeTime() 
   : C_general(), _identifierList(0) , _computeTime(FLT_MAX)
{

}

C_computeTime::C_computeTime(double& rv) 
   : C_general(), _identifierList(0) , _computeTime(rv)
{
   std::cout<< "the input value is " << rv << std::endl;
}

C_computeTime::C_computeTime(C_identifierList* identifierList) 
   : C_general(), _identifierList(identifierList) 
{

} 

C_computeTime::C_computeTime(const C_computeTime& rv) 
   : C_general(rv)
{
   copyOwnedHeap(rv);
}

C_computeTime& C_computeTime::operator=(const C_computeTime& rv)
{
   if (this != &rv) {
      C_general::operator=(rv);
      destructOwnedHeap();
      copyOwnedHeap(rv);
   }
   return *this;
}

void C_computeTime::duplicate(std::unique_ptr<C_computeTime>&& rv) const
{
   rv.reset(new C_computeTime(*this));
}

void C_computeTime::duplicate(std::unique_ptr<C_general>&& rv) const
{
   rv.reset(new C_computeTime(*this));
}

C_computeTime::~C_computeTime() 
{
   destructOwnedHeap();
}

void C_computeTime::copyOwnedHeap(const C_computeTime& rv)
{
   if (rv._identifierList) {
      std::unique_ptr<C_identifierList> dup;
      rv._identifierList->duplicate(std::move(dup));
      _identifierList = dup.release();
   } else {
      _identifierList = 0;
   }
}

void C_computeTime::destructOwnedHeap()
{
   delete _identifierList;
}

