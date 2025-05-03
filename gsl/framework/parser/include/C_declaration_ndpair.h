// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef C_declaration_ndpair_H
#define C_declaration_ndpair_H
#include "Copyright.h"

#include "C_declaration.h"
#include <memory>
#include <map>

class C_declarator;
class C_ndpair_clause;
class GslContext;
class SyntaxError;

class C_declaration_ndpair : public C_declaration
{
   public:
      C_declaration_ndpair(const C_declaration_ndpair&);
      C_declaration_ndpair(C_declarator *,  C_ndpair_clause *, SyntaxError *);
      virtual C_declaration_ndpair* duplicate() const;
      virtual ~C_declaration_ndpair();
      virtual void internalExecute(GslContext *);
      virtual void checkChildren();
      virtual void recursivePrint();

   private:
      C_declarator* _declarator;
      C_ndpair_clause* _ndp_clause;
};
#endif
