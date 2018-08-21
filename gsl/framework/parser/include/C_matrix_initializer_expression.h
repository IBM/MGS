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

#ifndef C_matrix_initializer_expression_H
#define C_matrix_initializer_expression_H
#include "Copyright.h"

#include "C_production_adi.h"

class ArrayDataItem;
class C_constant_list;
class C_int_constant_list;
class LensContext;
class SyntaxError;


class C_matrix_initializer_expression : public C_production_adi
{
   public:
      C_matrix_initializer_expression(const C_matrix_initializer_expression&);
      C_matrix_initializer_expression(C_int_constant_list *, SyntaxError *);
      virtual ~C_matrix_initializer_expression();
      virtual C_matrix_initializer_expression* duplicate() const;
      virtual void internalExecute(LensContext *, ArrayDataItem *);
      virtual void checkChildren();
      virtual void recursivePrint();
      const C_int_constant_list * getIntConstantList() const;
      int getOffset() {
	 return _offset;
      };

   private:
      C_int_constant_list* _intConstantList;
      int _offset;
};
#endif
