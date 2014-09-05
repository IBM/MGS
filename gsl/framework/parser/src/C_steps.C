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

#include "C_steps.h"
#include "C_int_constant_list.h"
#include "SyntaxError.h"
#include "C_production.h"

void C_steps::internalExecute(LensContext *c)
{
   _cIntConstList->execute(c);
}


C_steps::C_steps(const C_steps& rv)
   : C_production(rv), _cIntConstList(0)
{
   if (rv._cIntConstList) {
      _cIntConstList = rv._cIntConstList->duplicate();
   }
}


C_steps::C_steps(C_int_constant_list *c, SyntaxError * error)
   : C_production(error), _cIntConstList(c)
{
}


const std::list<int>* C_steps::getListInt() const
{
   return _cIntConstList->getList();
}


C_steps* C_steps::duplicate() const
{
   return new C_steps(*this);
}


C_steps::~C_steps()
{
   delete _cIntConstList;
}

void C_steps::checkChildren() 
{
   if (_cIntConstList) {
      _cIntConstList->checkChildren();
      if (_cIntConstList->isError()) {
         setError();
      }
   }
} 

void C_steps::recursivePrint() 
{
   if (_cIntConstList) {
      _cIntConstList->recursivePrint();
   }
   printErrorMessage();
} 
