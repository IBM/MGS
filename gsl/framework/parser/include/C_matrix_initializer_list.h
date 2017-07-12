// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BCM-YKT-11-19-2015
//
// (C) Copyright IBM Corp. 2005-2015  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#ifndef C_matrix_initializer_list_H
#define C_matrix_initializer_list_H
#include "Copyright.h"

#include "C_production_adi.h"

class C_default_clause;
class C_matrix_initializer_clause_list;
class LensContext;
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
      virtual void internalExecute(LensContext *, ArrayDataItem *);
      virtual void checkChildren();
      virtual void recursivePrint();
      C_default_clause* getDefaultClause() const;
      C_matrix_initializer_clause_list* getMatrixInitClauseList() const;

   private:
      C_default_clause* _defaultClause;
      C_matrix_initializer_clause_list* _matrixInitClauseList;

};
#endif
