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

#include "C_functor_specifier.h"
#include "LensContext.h"
#include "C_argument.h"
#include "C_declarator.h"
#include "C_argument_list.h"
#include "DataItem.h"
#include "FunctorDataItem.h"
#include "Functor.h"
#include "SyntaxError.h"
#include "SyntaxErrorException.h"
#include "C_production.h"


void C_functor_specifier::internalExecute(LensContext *c)
{
   _functorDeclarator->execute(c);
   _argumentList->execute(c);

   const DataItem* di = c->symTable.getEntry(_functorDeclarator->getName());
   if (di == 0) {
      std::string mes = _functorDeclarator->getName() + " not found";
      throwError(mes);
   }
   const FunctorDataItem* fdi = dynamic_cast<const FunctorDataItem*>(di);
   if (fdi == 0) {
      std::string mes = "dynamic cast of DataItem to FunctorDataItem failed";
      throwError(mes);
   }
   Functor* f = fdi->getFunctor();
   f->execute(c, *(_argumentList->getVectorDataItem()), _rval);
}


const DataItem* C_functor_specifier::getRVal() const
{
   return _rval.get();
}


C_functor_specifier::C_functor_specifier(const C_functor_specifier& rv)
   : C_production(rv), _functorDeclarator(0), _argumentList(0)
{
   if (rv._functorDeclarator) {
      _functorDeclarator = rv._functorDeclarator->duplicate();
   }
   if (rv._argumentList) {
      _argumentList = rv._argumentList->duplicate();
   }
   if (rv.getRVal()){
      rv._rval->duplicate(_rval);
   }
}


C_functor_specifier::C_functor_specifier(
   C_declarator *f, C_argument_list *a, SyntaxError * error)
   : C_production(error), _functorDeclarator(f), _argumentList(a)
{
}


C_functor_specifier* C_functor_specifier::duplicate() const
{
   return new C_functor_specifier(*this);
}


C_functor_specifier::~C_functor_specifier()
{
   delete _functorDeclarator;
   delete _argumentList;
}

void C_functor_specifier::checkChildren() 
{
   if (_functorDeclarator) {
      _functorDeclarator->checkChildren();
      if (_functorDeclarator->isError()) {
         setError();
      }
   }
   if (_argumentList) {
      _argumentList->checkChildren();
      if (_argumentList->isError()) {
         setError();
      }
   }
} 

void C_functor_specifier::recursivePrint() 
{
   if (_functorDeclarator) {
      _functorDeclarator->recursivePrint();
   }
   if (_argumentList) {
      _argumentList->recursivePrint();
   }
   printErrorMessage();
} 
