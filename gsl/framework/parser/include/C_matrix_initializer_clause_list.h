// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BMC-YKT-08-23-2011-2
//
// (C) Copyright IBM Corp. 2005-2014  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#ifndef C_matrix_initializer_clause_list_H
#define C_matrix_initializer_clause_list_H
#include "Copyright.h"

#include <list>
#include "C_production_adi.h"

class C_matrix_initializer_clause;
class LensContext;
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
      virtual void internalExecute(LensContext *, ArrayDataItem *);
      virtual void checkChildren();
      virtual void recursivePrint();
      std::list<C_matrix_initializer_clause>* getListMatrixInitClause() const;

   private:
      std::list<C_matrix_initializer_clause>* _listMatrixInitClause;

};
#endif
