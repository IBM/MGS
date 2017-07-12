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

#include "C_definition_functor.h"
#include "C_functor_definition.h"
#include "LensContext.h"
#include "FunctorTypeDataItem.h"
#include "ScriptFunctorTypeDataItem.h"
#include "SyntaxError.h"
#include "SyntaxErrorException.h"

void C_definition_functor::internalExecute(LensContext *c)
{
   // call executes
   _functor_def->execute(c);

   // put FunctorType into the symbol table

   FunctorTypeDataItem* ftdi;
   if (_functor_def->isScript()) {
      ftdi = new ScriptFunctorTypeDataItem;
   }
   else {
      ftdi = new FunctorTypeDataItem;
   }
   std::auto_ptr<DataItem> diap(ftdi);
   ftdi->setFunctorType(_functor_def->getFunctorType());
   ftdi->setCategory(_functor_def->getCategory());
   ftdi->setConstructorParams(_functor_def->getConstructorParams());
   ftdi->setFunctionParams(_functor_def->getFunctionParams());
   ftdi->setReturnParams(_functor_def->getReturnParams());
   std::string symbolName = _functor_def->getDeclarator();
   try {
      c->symTable.addEntry(symbolName, diap);
   } catch (SyntaxErrorException& e) {
      throwError("While defining functor, " + e.getError());
   }
}

C_definition_functor* C_definition_functor::duplicate() const
{
   return new C_definition_functor(*this);
}

C_definition_functor::C_definition_functor(const C_definition_functor& rv)
   : C_definition(rv), _functor_def(0)
{
   if (rv._functor_def) {
      _functor_def = rv._functor_def->duplicate();
   }
}

C_definition_functor::C_definition_functor(C_functor_definition *f, 
					   SyntaxError * error)
   : C_definition(error), _functor_def(f)
{
}

C_definition_functor::~C_definition_functor()
{
   delete _functor_def;
}

void C_definition_functor::checkChildren() 
{
   if (_functor_def) {
      _functor_def->checkChildren();
      if (_functor_def->isError()) {
         setError();
      }
   }
} 

void C_definition_functor::recursivePrint() 
{
   if (_functor_def) {
      _functor_def->recursivePrint();
   }
   printErrorMessage();
} 
