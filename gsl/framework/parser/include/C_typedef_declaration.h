// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef C_typedef_declaration_H
#define C_typedef_declaration_H
#include "Copyright.h"

#include "C_production.h"

class C_type_specifier;
class C_declarator;
class LensContext;
class SyntaxError;

class C_typedef_declaration : public C_production
{
   public:
      C_typedef_declaration(const C_typedef_declaration& rv);
      C_typedef_declaration(C_type_specifier *, C_declarator *, 
			    SyntaxError *);
      virtual ~C_typedef_declaration();
      virtual C_typedef_declaration* duplicate() const;
      virtual void internalExecute(LensContext *);
      virtual void checkChildren();
      virtual void recursivePrint();

   private:
      C_declarator* _id;
      C_type_specifier* _ts;
};
#endif
