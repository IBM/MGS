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

#ifndef C_matrix_init_declarator_H
#define C_matrix_init_declarator_H
#include "Copyright.h"

#include "C_production_adi.h"
#include "SyntaxError.h"

class C_declarator;
class C_int_constant_list;
class C_matrix_initializer;
class LensContext;
class ArrayDataItem;

class C_matrix_init_declarator : public C_production_adi
{
   public:
      C_matrix_init_declarator(const C_matrix_init_declarator&);
      C_matrix_init_declarator(C_declarator *, C_int_constant_list *, 
			       C_matrix_initializer *, SyntaxError *);
      virtual ~C_matrix_init_declarator ();
      virtual C_matrix_init_declarator* duplicate() const;
      virtual void internalExecute(LensContext *, ArrayDataItem *);
      virtual void checkChildren();
      virtual void recursivePrint();
      C_declarator* getDeclarator() const {
	 return _declarator;
      }
      C_int_constant_list* getIntConstantList() const {
	 return _intConstantList;
      }
      C_matrix_initializer* getMatrixInit() const {
	 return _matrixInit;
      }

   private:
      C_declarator* _declarator;
      C_int_constant_list* _intConstantList;
      C_matrix_initializer* _matrixInit;

};
#endif
