// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef C_dim_declaration_H
#define C_dim_declaration_H
#include "Copyright.h"

#include "C_production.h"

class C_int_constant_list;
class LensContext;

class C_dim_declaration: public C_production
{
   public:
      C_dim_declaration(const C_dim_declaration&);
      C_dim_declaration(C_int_constant_list *, SyntaxError *);
      virtual ~C_dim_declaration();
      virtual C_dim_declaration* duplicate() const;
      virtual void internalExecute(LensContext *);
      virtual void checkChildren();
      virtual void recursivePrint();
      const C_int_constant_list* getIntConstantList() const;

   private:
      C_int_constant_list* _intConstantList;
};
#endif
