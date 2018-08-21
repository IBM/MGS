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

#ifndef C_stride_H
#define C_stride_H
#include "Copyright.h"

#include <list>
#include "C_production.h"

class C_int_constant_list;
class LensContext;
class SyntaxError;

class C_stride : public C_production
{
   public:
      C_stride(const C_stride&);
      C_stride(C_int_constant_list *, SyntaxError *);
      virtual ~C_stride();
      virtual C_stride* duplicate() const;
      virtual void internalExecute(LensContext *);
      virtual void checkChildren();
      virtual void recursivePrint();
      const std::list<int> * getListInt() const;

   private:
      C_int_constant_list* _cIntConstList;
};
#endif
