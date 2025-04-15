// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef C_definition_type_H
#define C_definition_type_H
#include "Copyright.h"

#include "C_definition.h"

class C_type_definition;
class LensContext;
class SyntaxError;

class C_definition_type : public C_definition
{
   public:
      C_definition_type(const C_definition_type&);
      C_definition_type(C_type_definition *, SyntaxError *);
      virtual C_definition_type* duplicate() const;
      virtual ~C_definition_type();
      virtual void internalExecute(LensContext *);
      virtual void checkChildren();
      virtual void recursivePrint();

   private:
      C_type_definition* _typeDefinition;
};
#endif
