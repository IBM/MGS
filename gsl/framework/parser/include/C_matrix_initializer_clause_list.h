// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef C_matrix_initializer_clause_list_H
#define C_matrix_initializer_clause_list_H
#include "Copyright.h"

#include <list>
#include "C_production_adi.h"

class C_matrix_initializer_clause;
class GslContext;
class ArrayDataItem;
class SyntaxError;

class C_matrix_initializer_clause_list : public C_production_adi
{
   public:
      C_matrix_initializer_clause_list(
	 const C_matrix_initializer_clause_list&);
      C_matrix_initializer_clause_list(
	 C_matrix_initializer_clause *, SyntaxError * error);
      C_matrix_initializer_clause_list(
	 C_matrix_initializer_clause_list *, C_matrix_initializer_clause *, 
	 SyntaxError *);
      std::list<C_matrix_initializer_clause>* releaseList();
      virtual ~C_matrix_initializer_clause_list();
      virtual C_matrix_initializer_clause_list* duplicate() const;
      virtual void internalExecute(GslContext *, ArrayDataItem *);
      virtual void checkChildren();
      virtual void recursivePrint();
      std::list<C_matrix_initializer_clause>* getListMatrixInitClause() const;

   private:
      std::list<C_matrix_initializer_clause>* _listMatrixInitClause;

};
#endif
