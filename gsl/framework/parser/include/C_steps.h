// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef C_steps_H
#define C_steps_H
#include "Copyright.h"

#include <list>
#include "C_production.h"

class C_int_constant_list;
class GslContext;
class SyntaxError;

class C_steps : public C_production
{
   public:
      C_steps(const C_steps&);
      C_steps(C_int_constant_list *, SyntaxError *);
      virtual ~C_steps();
      virtual C_steps* duplicate() const;
      virtual void internalExecute(GslContext *);
      virtual void checkChildren();
      virtual void recursivePrint();
      const std::list<int>* getListInt() const;

   private:
      C_int_constant_list* _cIntConstList;
};
#endif
