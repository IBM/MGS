// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef C_declaration_trigger_H
#define C_declaration_trigger_H
#include "Copyright.h"

#include "C_declaration.h"
#include "GslContext.h"
#include <string>
#include <memory>
#include <map>

class C_declarator;
class C_trigger;
class SyntaxError;

class C_declaration_trigger : public C_declaration
{
   public:
      C_declaration_trigger(const C_declaration_trigger&);
      C_declaration_trigger(C_declarator *, C_trigger *, SyntaxError *);
      virtual C_declaration_trigger* duplicate() const;
      virtual ~C_declaration_trigger();
      virtual void internalExecute(GslContext *);
      virtual void checkChildren();
      virtual void recursivePrint();

   private:
      C_declarator* _declarator;
      C_trigger* _trigger;
      std::string* _name;
};
#endif
