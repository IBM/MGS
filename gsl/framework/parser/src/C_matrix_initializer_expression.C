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

#include "C_matrix_initializer_expression.h"
#include "ArrayDataItem.h"
#include "C_constant_list.h"
#include "C_int_constant_list.h"
#include "SyntaxError.h"
#include "C_production_adi.h"

void C_matrix_initializer_expression::internalExecute(
   LensContext *c, ArrayDataItem *adi)
{
    _intConstantList->execute(c, adi);
    _offset = _intConstantList->getOffset();
}


C_matrix_initializer_expression::C_matrix_initializer_expression(
   C_int_constant_list *icl, SyntaxError * error)
   : C_production_adi(error), _intConstantList(icl), _offset(0)
{
}


C_matrix_initializer_expression::C_matrix_initializer_expression(
   const C_matrix_initializer_expression& rv)
   : C_production_adi(rv), _intConstantList(0), _offset(rv._offset)
{
   if (rv._intConstantList) {
      _intConstantList = rv._intConstantList->duplicate();
   }
}


C_matrix_initializer_expression* 
C_matrix_initializer_expression::duplicate() const
{
   return new C_matrix_initializer_expression(*this);
}


C_matrix_initializer_expression::~C_matrix_initializer_expression()
{
   delete _intConstantList;
}


const C_int_constant_list *
C_matrix_initializer_expression::getIntConstantList() const
{

   return _intConstantList;

}

void C_matrix_initializer_expression::checkChildren() 
{
   if (_intConstantList) {
      _intConstantList->checkChildren();
      if (_intConstantList->isError()) {
         setError();
      }
   }
} 

void C_matrix_initializer_expression::recursivePrint() 
{
   if (_intConstantList) {
      _intConstantList->recursivePrint();
   }
   printErrorMessage();
} 
