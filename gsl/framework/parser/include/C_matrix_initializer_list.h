// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef C_matrix_initializer_list_H
#define C_matrix_initializer_list_H
#include "Copyright.h"

#include "C_production_adi.h"

class C_default_clause;
class C_matrix_initializer_clause_list;
class GslContext;
class ArrayDataItem;
class SyntaxError;


class C_matrix_initializer_list : public C_production_adi
{
   public:
      C_matrix_initializer_list(const C_matrix_initializer_list&);
      C_matrix_initializer_list(C_default_clause *, SyntaxError *);
      C_matrix_initializer_list(C_default_clause *, 
				C_matrix_initializer_clause_list *, 
				SyntaxError *);
      C_matrix_initializer_list(C_matrix_initializer_clause_list *, 
				SyntaxError *);
      virtual ~C_matrix_initializer_list ();
      virtual C_matrix_initializer_list* duplicate() const;
      virtual void internalExecute(GslContext *, ArrayDataItem *);
      virtual void checkChildren();
      virtual void recursivePrint();
      C_default_clause* getDefaultClause() const;
      C_matrix_initializer_clause_list* getMatrixInitClauseList() const;

   private:
      C_default_clause* _defaultClause;
      C_matrix_initializer_clause_list* _matrixInitClauseList;

};
#endif
