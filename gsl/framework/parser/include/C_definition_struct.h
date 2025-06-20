// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef C_DEFINITION_STRUCT_H
#define C_DEFINITION_STRUCT_H
#include "Copyright.h"

#include <string>
#include <list>

#include "C_definition.h"

class C_declarator;
class C_parameter_type;
class C_parameter_type_list;
class GslContext;
class InstanceFactory;
class SyntaxError;

class C_definition_struct : public C_definition
{
   public:
      C_definition_struct(const C_definition_struct&);
      C_definition_struct(C_declarator *, SyntaxError *);
      virtual ~C_definition_struct();
      virtual C_definition_struct* duplicate() const;
      virtual void internalExecute(GslContext *);
      virtual void checkChildren();
      virtual void recursivePrint();

      // accessor methods
      std::string getDeclarator();
      InstanceFactory* getInstanceFactory();

   private:
      C_declarator* _declarator;
      InstanceFactory* _instanceFactory;
};
#endif
