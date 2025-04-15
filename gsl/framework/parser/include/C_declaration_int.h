// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef C_declaration_int_H
#define C_declaration_int_H
#include "Copyright.h"

#include "C_declaration.h"
#include <memory>
#include <map>

class C_declarator;
class LensContext;
class SyntaxError;

class C_declaration_int : public C_declaration
{
   public:
      C_declaration_int(const C_declaration_int&);
      C_declaration_int(C_declarator *, int, SyntaxError *);
      virtual C_declaration_int* duplicate() const;
      virtual ~C_declaration_int();
      virtual void internalExecute(LensContext *);
      virtual void checkChildren();
      virtual void recursivePrint();

   private:
      C_declarator* _declarator;
      int _intValue;
};
#endif
