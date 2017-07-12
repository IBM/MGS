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

#ifndef C_matrix_initializer_H
#define C_matrix_initializer_H
#include "Copyright.h"

#include "C_production_adi.h"

class C_matrix_initializer_list;
class LensContext;
class ArrayDataItem;
class SyntaxError;

class C_matrix_initializer : public C_production_adi
{
   public:
      C_matrix_initializer(const C_matrix_initializer&);
      C_matrix_initializer(C_matrix_initializer_list *, SyntaxError *);
      virtual ~C_matrix_initializer();
      virtual C_matrix_initializer* duplicate() const;
      virtual void internalExecute(LensContext *, ArrayDataItem *);
      virtual void checkChildren();
      virtual void recursivePrint();
      C_matrix_initializer_list* getMatrixInitList() const {
	 return _matrixInitList;
      }

   private:
      C_matrix_initializer_list* _matrixInitList;

};
#endif
