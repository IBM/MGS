// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "C_matrix_initializer_clause_list.h"
#include "ArrayDataItem.h"
#include "C_matrix_initializer_clause.h"
#include "SyntaxError.h"
#include "C_production_adi.h"

void C_matrix_initializer_clause_list::internalExecute(LensContext *c, ArrayDataItem *adi)
{
   std::list<C_matrix_initializer_clause>::iterator iter, 
      end = _listMatrixInitClause->end();
   for(iter = _listMatrixInitClause->begin(); iter != end; ++iter) {
      iter->execute(c, adi);
   }
}


C_matrix_initializer_clause_list::C_matrix_initializer_clause_list(
   const C_matrix_initializer_clause_list& rv)
   : C_production_adi(rv), _listMatrixInitClause(0)
{
   if (rv._listMatrixInitClause) {
      _listMatrixInitClause = new std::list<C_matrix_initializer_clause>(
	 *(rv._listMatrixInitClause));
   }
}


C_matrix_initializer_clause_list::C_matrix_initializer_clause_list(
   C_matrix_initializer_clause *mic, SyntaxError * error)
   : C_production_adi(error), _listMatrixInitClause(0)
{
   _listMatrixInitClause = new std::list<C_matrix_initializer_clause>;
   if (mic) {
      _listMatrixInitClause->push_back(*mic);
      delete mic;
   }
}


C_matrix_initializer_clause_list::C_matrix_initializer_clause_list(
   C_matrix_initializer_clause_list *micl, C_matrix_initializer_clause *mic, 
   SyntaxError * error)
   : C_production_adi(error), _listMatrixInitClause(0)
{
   if (micl) {
      if (micl->isError()) {
	 delete _error;
	 _error = micl->_error->duplicate();
      }
      _listMatrixInitClause = micl->releaseList();
      if (mic) _listMatrixInitClause->push_back(*mic);
   }
   delete micl;
   delete mic;
}


std::list<C_matrix_initializer_clause>*
C_matrix_initializer_clause_list::releaseList()
{
   std::list<C_matrix_initializer_clause> *retval = _listMatrixInitClause;
   _listMatrixInitClause = 0;
   return retval;
}


C_matrix_initializer_clause_list* 
C_matrix_initializer_clause_list::duplicate() const
{
   return new C_matrix_initializer_clause_list(*this);
}


C_matrix_initializer_clause_list::~C_matrix_initializer_clause_list()
{
   delete _listMatrixInitClause;
}


std::list<C_matrix_initializer_clause>* 
C_matrix_initializer_clause_list::getListMatrixInitClause() const
{
   return _listMatrixInitClause;
}

void C_matrix_initializer_clause_list::checkChildren() 
{
   if (_listMatrixInitClause) {
      std::list<C_matrix_initializer_clause>::iterator i, begin, end;
      begin =_listMatrixInitClause->begin();
      end =_listMatrixInitClause->end();
      for(i=begin;i!=end;++i) {
	 i->checkChildren();
	 if (i->isError()) {
	    setError();
	 }
      }
   }
} 

void C_matrix_initializer_clause_list::recursivePrint() 
{
   if (_listMatrixInitClause) {
      std::list<C_matrix_initializer_clause>::iterator i, begin, end;
      begin =_listMatrixInitClause->begin();
      end =_listMatrixInitClause->end();
      for(i=begin;i!=end;++i) {
	 i->recursivePrint();
      }
   }
   printErrorMessage();
} 
