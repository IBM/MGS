// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef C_declaration_typedef_H
#define C_declaration_typedef_H
#include "Copyright.h"

#include "C_declaration.h"

class C_typedef_declaration;
class GslContext;
class SyntaxError;

class C_declaration_typedef : public C_declaration
{
   public:
      C_declaration_typedef(const C_declaration_typedef&);
      C_declaration_typedef(C_typedef_declaration *, SyntaxError *);
      virtual C_declaration_typedef* duplicate() const;
      virtual ~C_declaration_typedef();
      virtual void internalExecute(GslContext *);
      virtual void checkChildren();
      virtual void recursivePrint();

   private:
      C_typedef_declaration* _t;
};
#endif
