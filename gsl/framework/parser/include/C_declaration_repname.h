// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef C_declaration_repname_H
#define C_declaration_repname_H
#include "Copyright.h"

#include "C_declaration.h"

class C_declarator;
class C_repname;
class LensContext;

class C_declaration_repname : public C_declaration
{
   public:
      C_declaration_repname(const C_declaration_repname&);
      C_declaration_repname(C_declarator *, C_repname *, SyntaxError *);
      virtual C_declaration_repname* duplicate() const;
      virtual ~C_declaration_repname();
      virtual void internalExecute(LensContext *);
      virtual void checkChildren();
      virtual void recursivePrint();

   private:
      C_declarator *_declarator;
      C_repname *_repname;

};
#endif
