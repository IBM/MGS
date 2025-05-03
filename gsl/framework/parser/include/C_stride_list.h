// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef C_stride_list_H
#define C_stride_list_H
#include "Copyright.h"

#include "StridesList.h"
#include "C_production.h"

class C_order;
class C_steps;
class C_stride;
class GslContext;
class StrideList;
class SyntaxError;

class C_stride_list : public C_production, public StridesList
{
   public:
      C_stride_list(const C_stride_list&);
      C_stride_list(C_steps *, C_stride *,C_order *, SyntaxError *);
      virtual ~C_stride_list();
      virtual C_stride_list* duplicate() const;
      virtual void internalExecute(GslContext *);
      virtual void checkChildren();
      virtual void recursivePrint();
      const std::list<int>& getStepsListInt() const;
      const std::list<int>& getStrideListInt() const;
      const std::list<int>& getOrderListInt() const;
      StridesList* getStridesList();

   private:
      C_steps* _cSteps;
      C_stride* _cStride;
      C_order* _cOrder;
};
#endif
