// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef C_matrix_initializer_H
#define C_matrix_initializer_H
#include "Copyright.h"

#include "C_production_adi.h"

class C_matrix_initializer_list;
class GslContext;
class ArrayDataItem;
class SyntaxError;

class C_matrix_initializer : public C_production_adi
{
   public:
      C_matrix_initializer(const C_matrix_initializer&);
      C_matrix_initializer(C_matrix_initializer_list *, SyntaxError *);
      virtual ~C_matrix_initializer();
      virtual C_matrix_initializer* duplicate() const;
      virtual void internalExecute(GslContext *, ArrayDataItem *);
      virtual void checkChildren();
      virtual void recursivePrint();
      C_matrix_initializer_list* getMatrixInitList() const {
	 return _matrixInitList;
      }

   private:
      C_matrix_initializer_list* _matrixInitList;

};
#endif
