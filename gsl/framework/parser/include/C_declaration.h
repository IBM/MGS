// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef C_declaration_H
#define C_declaration_H
#include "Copyright.h"

#include "C_production.h"

class GslContext;

class C_declaration : public C_production
{
   public:
      C_declaration(SyntaxError* error);
      C_declaration(const C_declaration&);
      virtual ~C_declaration();
      virtual C_declaration* duplicate() const = 0;
      virtual void internalExecute(GslContext *);
      virtual void checkChildren();
      virtual void recursivePrint();
};
#endif
