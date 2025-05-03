// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef C_functor_specifier_H
#define C_functor_specifier_H
#include "Copyright.h"

#include <memory>
#include "C_production.h"

class C_declarator;
class C_argument_list;
class GslContext;
class DataItem;
class SyntaxError;

class C_functor_specifier : public C_production
{
   public:
      C_functor_specifier(const C_functor_specifier&);
      C_functor_specifier(C_declarator *, C_argument_list *, SyntaxError *);
      virtual ~C_functor_specifier();
      virtual C_functor_specifier* duplicate() const;
      virtual void internalExecute(GslContext *);
      virtual void checkChildren();
      virtual void recursivePrint();

      // Will return null if void
      const DataItem* getRVal() const;

   private:
      C_declarator* _functorDeclarator;
      C_argument_list* _argumentList;
      std::unique_ptr<DataItem> _rval;
};
#endif
