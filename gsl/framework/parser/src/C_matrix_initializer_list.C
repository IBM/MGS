// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "C_matrix_initializer_list.h"
#include "C_default_clause.h"
#include "C_matrix_initializer_clause_list.h"
#include "ArrayDataItem.h"
#include "IntArrayDataItem.h"
#include "FloatArrayDataItem.h"
#include "SyntaxError.h"
#include "C_production_adi.h"

void C_matrix_initializer_list::internalExecute(LensContext *c, 
						ArrayDataItem *adi)
{

   if (_defaultClause)
      _defaultClause->execute(c, adi);
   else {// set everything to zero
      if (IntArrayDataItem *I_adi = dynamic_cast<IntArrayDataItem*>(adi))
         fill(I_adi->getModifiableIntVector()->begin(), 
	      I_adi->getModifiableIntVector()->end(), 0);
      else {
         FloatArrayDataItem *F_adi = dynamic_cast<FloatArrayDataItem*>(adi);
         fill (F_adi->getModifiableFloatVector()->begin(), 
	       F_adi->getModifiableFloatVector()->end(), 0.0);
      }
   }
   if(_matrixInitClauseList) {
      _matrixInitClauseList->execute(c, adi);
   }
}


C_matrix_initializer_list::C_matrix_initializer_list(
   const C_matrix_initializer_list& rv)
   : C_production_adi(rv), _defaultClause(0), _matrixInitClauseList(0)
{
   if (rv._defaultClause) {
      _defaultClause = rv._defaultClause->duplicate();
   }
   if (rv._matrixInitClauseList) {
      _matrixInitClauseList = rv._matrixInitClauseList->duplicate();
   }
}


C_matrix_initializer_list::C_matrix_initializer_list(C_default_clause *dc, 
						     SyntaxError * error)
   : C_production_adi(error), _defaultClause(dc), _matrixInitClauseList(0)
{
   _defaultClause = dc;
}


C_matrix_initializer_list::C_matrix_initializer_list(
   C_default_clause *dc, C_matrix_initializer_clause_list *micl, 
   SyntaxError * error)
   : C_production_adi(error), _defaultClause(dc), _matrixInitClauseList(micl) 

{
}


C_matrix_initializer_list::C_matrix_initializer_list(
   C_matrix_initializer_clause_list *micl, SyntaxError * error)
   : C_production_adi(error), _defaultClause(0), _matrixInitClauseList(micl) 
{
}


C_matrix_initializer_list* C_matrix_initializer_list::duplicate() const
{
   return new C_matrix_initializer_list(*this);
}


C_matrix_initializer_list::~C_matrix_initializer_list()
{
   delete _defaultClause;
   delete _matrixInitClauseList;
}


C_default_clause * C_matrix_initializer_list::getDefaultClause() const
{
   return _defaultClause;
}


C_matrix_initializer_clause_list * 
C_matrix_initializer_list::getMatrixInitClauseList() const
{
   return _matrixInitClauseList;
}

void C_matrix_initializer_list::checkChildren() 
{
   if (_defaultClause) {
      _defaultClause->checkChildren();
      if (_defaultClause->isError()) {
         setError();
      }
   }
   if (_matrixInitClauseList) {
      _matrixInitClauseList->checkChildren();
      if (_matrixInitClauseList->isError()) {
         setError();
      }
   }
} 

void C_matrix_initializer_list::recursivePrint() 
{
   if (_defaultClause) {
      _defaultClause->recursivePrint();
   }
   if (_matrixInitClauseList) {
      _matrixInitClauseList->recursivePrint();
   }
   printErrorMessage();
} 
