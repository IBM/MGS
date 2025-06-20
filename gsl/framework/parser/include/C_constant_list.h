// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef C_constant_list_H
#define C_constant_list_H
#include "Copyright.h"

#include <list>
#include "C_production.h"

class C_constant;
class GslContext;
class SyntaxError;

class C_constant_list : public C_production
{
   public:
      C_constant_list(const C_constant_list&);
      C_constant_list(C_constant *, SyntaxError *);
      C_constant_list(C_constant_list *, C_constant *, SyntaxError *);
      virtual C_constant_list* duplicate() const;
      std::list<C_constant>* releaseList();
      virtual void internalExecute(GslContext *);
      virtual void checkChildren();
      virtual void recursivePrint();
      virtual ~C_constant_list();
      std::list<C_constant>* getList()const;

   private:
      std::list<C_constant>* _list;
};
#endif
