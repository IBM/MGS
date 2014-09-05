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

#include "C_stride.h"
#include "C_int_constant_list.h"
#include "SyntaxError.h"
#include "C_production.h"

void C_stride::internalExecute(LensContext *c)
{
   _cIntConstList->execute(c);
}


C_stride::C_stride(const C_stride& rv)
   : C_production(rv), _cIntConstList(0)
{
   if (rv._cIntConstList) {
      _cIntConstList = rv._cIntConstList->duplicate();
   }
}


C_stride::C_stride(C_int_constant_list *c, SyntaxError * error)
   : C_production(error), _cIntConstList(c)
{
}


const std::list<int> * C_stride::getListInt() const
{
   return _cIntConstList->getList();
}


C_stride* C_stride::duplicate() const
{
   return new C_stride(*this);
}


C_stride::~C_stride()
{
   delete _cIntConstList;
}

void C_stride::checkChildren() 
{
   if (_cIntConstList) {
      _cIntConstList->checkChildren();
      if (_cIntConstList->isError()) {
         setError();
      }
   }
} 

void C_stride::recursivePrint() 
{
   if (_cIntConstList) {
      _cIntConstList->recursivePrint();
   }
   printErrorMessage();
} 
