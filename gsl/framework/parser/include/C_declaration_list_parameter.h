// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef C_declaration_list_parameter_H
#define C_declaration_list_parameter_H
#include "Copyright.h"

#include <memory>
#include <map>

#include "C_declaration.h"
class C_list_parameter_type;
class C_declarator;
class C_type_specifier;
class C_argument_list;
class LensContext;
class SyntaxError;

class C_declaration_list_parameter : public C_declaration
{
   public:
      C_declaration_list_parameter(const C_declaration_list_parameter&);
      C_declaration_list_parameter(C_type_specifier *, C_declarator *, 
				   C_argument_list *, SyntaxError *);
      C_declaration_list_parameter(C_declarator *, C_argument_list *, 
				   SyntaxError *);
      virtual C_declaration_list_parameter* duplicate() const;
      virtual ~C_declaration_list_parameter();
      virtual void internalExecute(LensContext *);
      virtual void checkChildren();
      virtual void recursivePrint();

   private:
      C_type_specifier* _typeSpec;
      C_declarator* _declarator;
      C_argument_list* _argumentList;

};
#endif
