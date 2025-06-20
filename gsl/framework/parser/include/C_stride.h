// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef C_stride_H
#define C_stride_H
#include "Copyright.h"

#include <list>
#include "C_production.h"

class C_int_constant_list;
class GslContext;
class SyntaxError;

class C_stride : public C_production
{
   public:
      C_stride(const C_stride&);
      C_stride(C_int_constant_list *, SyntaxError *);
      virtual ~C_stride();
      virtual C_stride* duplicate() const;
      virtual void internalExecute(GslContext *);
      virtual void checkChildren();
      virtual void recursivePrint();
      const std::list<int> * getListInt() const;

   private:
      C_int_constant_list* _cIntConstList;
};
#endif
