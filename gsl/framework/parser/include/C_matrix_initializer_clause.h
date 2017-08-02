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

#ifndef C_matrix_initializer_clause_H
#define C_matrix_initializer_clause_H
#include "Copyright.h"

#include "C_production_adi.h"

class C_matrix_initializer_expression;
class C_constant_list;
class LensContext;
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
      virtual void internalExecute(LensContext *, ArrayDataItem *);
      virtual void checkChildren();
      virtual void recursivePrint();
      C_matrix_initializer_expression* getMatrixInitExp() const;
      C_constant_list* getConstantList() const;

   private:
      C_matrix_initializer_expression* _matrixInitExp;
      C_constant_list* _constantList;
};
#endif
