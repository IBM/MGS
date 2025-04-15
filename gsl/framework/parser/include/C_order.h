// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef C_order_H
#define C_order_H
#include "Copyright.h"

#include <list>
#include "C_production.h"

class C_int_constant_list;
class LensContext;
class SyntaxError;

class C_order : public C_production
{
   public:
      C_order(const C_order&);
      C_order(C_int_constant_list *, SyntaxError * error);
      virtual ~C_order();
      virtual C_order* duplicate() const;
      virtual void internalExecute(LensContext *);
      virtual void checkChildren();
      virtual void recursivePrint();
      const std::list<int>* getListInt() const;

   private:
      C_int_constant_list*_cIntConstList;
};
#endif
