// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "C_matrix_init_declarator.h"
#include "C_declarator.h"
#include "C_matrix_initializer.h"
#include "C_int_constant_list.h"
#include "ArrayDataItem.h"
#include "SyntaxError.h"
#include "C_production_adi.h"

void C_matrix_init_declarator::internalExecute(LensContext *c, ArrayDataItem *di)
{
   _declarator->execute(c);

   // the dimensions have been already set, do not need this
   // _intConstantList->execute(c);

   _matrixInit->execute(c, di);
}


C_matrix_init_declarator::C_matrix_init_declarator(
   const C_matrix_init_declarator& rv)
   : C_production_adi(rv), _declarator(0), _intConstantList(0), _matrixInit(0)
{
   if (rv._declarator) {
      _declarator = rv._declarator->duplicate();
   }
   if (rv._intConstantList) {
      _intConstantList = rv._intConstantList->duplicate();
   }
   if (rv._matrixInit) {
      _matrixInit  = rv._matrixInit->duplicate();
   }
}


C_matrix_init_declarator::C_matrix_init_declarator(
   C_declarator *d, C_int_constant_list *icl, C_matrix_initializer *mi, 
   SyntaxError * error)
   : C_production_adi(error), _declarator(d), _intConstantList(icl), 
     _matrixInit(mi)
{
}


C_matrix_init_declarator* C_matrix_init_declarator::duplicate() const
{
   return new C_matrix_init_declarator(*this);
}


C_matrix_init_declarator::~C_matrix_init_declarator()
{

   delete _declarator;
   delete _intConstantList;
   delete _matrixInit;

}

void C_matrix_init_declarator::checkChildren() 
{
   if (_declarator) {
      _declarator->checkChildren();
      if (_declarator->isError()) {
         setError();
      }
   }
   if (_intConstantList) {
      _intConstantList->checkChildren();
      if (_intConstantList->isError()) {
         setError();
      }
   }
   if (_matrixInit) {
      _matrixInit->checkChildren();
      if (_matrixInit->isError()) {
         setError();
      }
   }
} 

void C_matrix_init_declarator::recursivePrint() 
{
   if (_declarator) {
      _declarator->recursivePrint();
   }
   if (_intConstantList) {
      _intConstantList->recursivePrint();
   }
   if (_matrixInit) {
      _matrixInit->recursivePrint();
   }
   printErrorMessage();
} 
