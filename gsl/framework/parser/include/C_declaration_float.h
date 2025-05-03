// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef C_declaration_float_H
#define C_declaration_float_H
#include "Copyright.h"

#include "C_declaration.h"

#include <memory>
#include <map>

class C_constant;
class C_declarator;
class GslContext;
class SyntaxError;

class C_declaration_float : public C_declaration
{
   public:
      C_declaration_float(C_declarator *, C_constant *, SyntaxError *);
      C_declaration_float(const C_declaration_float&);
      virtual C_declaration_float* duplicate() const;
      virtual ~C_declaration_float();
      virtual void internalExecute(GslContext *);
      virtual void checkChildren();
      virtual void recursivePrint();

   private:
      C_constant* _constant;
      C_declarator* _declarator;
      float _floatValue;

};
#endif
