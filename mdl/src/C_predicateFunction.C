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

#include "C_predicateFunction.h"
#include "C_general.h"
#include "C_generalList.h"
#include "MdlContext.h"
#include "InternalException.h"
#include "PredicateFunction.h"
#include "C_identifierList.h"

#include <memory>
#include <string>

void C_predicateFunction::execute(MdlContext* context) 
{
   if (_identifierList == 0) {
      throw InternalException(
	 "_identifierList is 0 in C_predicateFunction::execute");
   }
   _identifierList->execute(context);
}

void C_predicateFunction::addToList(C_generalList* gl) 
{
   const std::vector<std::string>& ids = _identifierList->getIdentifiers();
   std::vector<std::string>::const_iterator it, end = ids.end();
   for (it = ids.begin(); it != end; ++ it) {
      std::auto_ptr<PredicateFunction> predicateFunction(
	 new PredicateFunction(*it));
      gl->addPredicateFunction(predicateFunction);
   }
}


C_predicateFunction::C_predicateFunction() 
   : C_general(), _identifierList(0) 
{

}

C_predicateFunction::C_predicateFunction(C_identifierList* identifierList) 
   : C_general(), _identifierList(identifierList) 
{

} 

C_predicateFunction::C_predicateFunction(const C_predicateFunction& rv) 
   : C_general(rv)
{
   copyOwnedHeap(rv);
}

C_predicateFunction& C_predicateFunction::operator=(const C_predicateFunction& rv)
{
   if (this != &rv) {
      C_general::operator=(rv);
      destructOwnedHeap();
      copyOwnedHeap(rv);
   }
   return *this;
}

void C_predicateFunction::duplicate(std::auto_ptr<C_predicateFunction>& rv) const
{
   rv.reset(new C_predicateFunction(*this));
}

void C_predicateFunction::duplicate(std::auto_ptr<C_general>& rv) const
{
   rv.reset(new C_predicateFunction(*this));
}

C_predicateFunction::~C_predicateFunction() 
{
   destructOwnedHeap();
}

void C_predicateFunction::copyOwnedHeap(const C_predicateFunction& rv)
{
   if (rv._identifierList) {
      std::auto_ptr<C_identifierList> dup;
      rv._identifierList->duplicate(dup);
      _identifierList = dup.release();
   } else {
      _identifierList = 0;
   }
}

void C_predicateFunction::destructOwnedHeap()
{
   delete _identifierList;
}

