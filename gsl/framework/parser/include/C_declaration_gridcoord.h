// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef C_declaration_gridcoord_H
#define C_declaration_gridcoord_H
#include "Copyright.h"

#include "C_declaration.h"

class C_declarator;
class C_gridset;
class GslContext;
class SyntaxError;

class C_declaration_gridcoord : public C_declaration
{
   public:
      C_declaration_gridcoord(const C_declaration_gridcoord&);
      C_declaration_gridcoord(C_declarator *, C_gridset *, SyntaxError *);
      virtual C_declaration_gridcoord* duplicate() const;
      virtual ~C_declaration_gridcoord();
      virtual void internalExecute(GslContext *);
      virtual void checkChildren();
      virtual void recursivePrint();

   private:
      C_declarator* _declarator;
      C_gridset* _gridset;
};
#endif
