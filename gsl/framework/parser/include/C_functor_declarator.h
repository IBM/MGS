// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef C_functor_declarator_H
#define C_functor_declarator_H
#include "Copyright.h"

#include <string>
#include "C_production.h"

class GslContext;
class SyntaxError;

class C_functor_declarator : public C_production
{
   public:
      C_functor_declarator(const C_functor_declarator&);
      C_functor_declarator(const std::string&, SyntaxError *);
      virtual ~C_functor_declarator();
      virtual C_functor_declarator* duplicate() const;
      virtual void internalExecute(GslContext *);
      virtual void checkChildren();
      virtual void recursivePrint();

   private:
      std::string _id;
};
#endif
