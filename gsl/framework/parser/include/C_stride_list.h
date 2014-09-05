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

#ifndef C_stride_list_H
#define C_stride_list_H
#include "Copyright.h"

#include "StridesList.h"
#include "C_production.h"

class C_order;
class C_steps;
class C_stride;
class LensContext;
class StrideList;
class SyntaxError;

class C_stride_list : public C_production, public StridesList
{
   public:
      C_stride_list(const C_stride_list&);
      C_stride_list(C_steps *, C_stride *,C_order *, SyntaxError *);
      virtual ~C_stride_list();
      virtual C_stride_list* duplicate() const;
      virtual void internalExecute(LensContext *);
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
