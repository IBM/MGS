// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef C_matrix_initializer_clause_H
#define C_matrix_initializer_clause_H
#include "Copyright.h"

#include "C_production_adi.h"

class C_matrix_initializer_expression;
class C_constant_list;
class GslContext;
class ArrayDataItem;
class SyntaxError;

class C_matrix_initializer_clause : public C_production_adi
{
   public:
      C_matrix_initializer_clause(const C_matrix_initializer_clause&);
      C_matrix_initializer_clause(C_matrix_initializer_expression *, 
				  C_constant_list *, 
				  SyntaxError *);
      C_matrix_initializer_clause(C_constant_list *, SyntaxError * error);
      virtual ~C_matrix_initializer_clause ();
      virtual C_matrix_initializer_clause* duplicate() const;
      virtual void internalExecute(GslContext *, ArrayDataItem *);
      virtual void checkChildren();
      virtual void recursivePrint();
      C_matrix_initializer_expression* getMatrixInitExp() const;
      C_constant_list* getConstantList() const;

   private:
      C_matrix_initializer_expression* _matrixInitExp;
      C_constant_list* _constantList;
};
#endif
