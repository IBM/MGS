// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "C_matrix_initializer.h"
#include "C_matrix_initializer_list.h"
#include "ArrayDataItem.h"
#include "SyntaxError.h"
#include "C_production_adi.h"

void C_matrix_initializer::internalExecute(LensContext *c, ArrayDataItem *adi)
{
   _matrixInitList->execute(c, adi);
}

C_matrix_initializer::C_matrix_initializer(const C_matrix_initializer& rv)
   : C_production_adi(rv), _matrixInitList(0)
{
   if (rv._matrixInitList) {
      _matrixInitList = rv._matrixInitList->duplicate();
   }
}

C_matrix_initializer::C_matrix_initializer(C_matrix_initializer_list *mil, 
					   SyntaxError * error)
   : C_production_adi(error), _matrixInitList(mil)
{
}

C_matrix_initializer* C_matrix_initializer::duplicate() const
{
   return new C_matrix_initializer(*this);
}

C_matrix_initializer::~C_matrix_initializer()
{
   delete _matrixInitList;
}

void C_matrix_initializer::checkChildren() 
{
   if (_matrixInitList) {
      _matrixInitList->checkChildren();
      if (_matrixInitList->isError()) {
         setError();
      }
   }
} 

void C_matrix_initializer::recursivePrint() 
{
   if (_matrixInitList) {
      _matrixInitList->recursivePrint();
   }
   printErrorMessage();
} 
