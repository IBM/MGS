// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef C_DEFINITION_FUNCTOR_H
#define C_DEFINITION_FUNCTOR_H
#include "Copyright.h"

#include <memory>
#include <map>
#include "C_definition.h"

class C_functor_definition;
class GslContext;
class SyntaxError;

class C_definition_functor : public C_definition
{
   public:
      C_definition_functor(const C_definition_functor&);
      C_definition_functor(C_functor_definition *, SyntaxError *);
      virtual C_definition_functor* duplicate() const;
      virtual ~C_definition_functor();
      virtual void internalExecute(GslContext *);
      virtual void checkChildren();
      virtual void recursivePrint();

   private:
      C_functor_definition* _functor_def;
};
#endif
