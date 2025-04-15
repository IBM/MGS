// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef C_declaration_index_set_H
#define C_declaration_index_set_H
#include "Copyright.h"

#include "C_declaration.h"
#include <memory>
#include <map>

class C_declarator;
class C_index_set;
class LensContext;
class SyntaxError;

class C_declaration_index_set : public C_declaration
{
   public:
      C_declaration_index_set(const C_declaration_index_set&);
      C_declaration_index_set(C_declarator *,  C_index_set *, SyntaxError *);
      virtual C_declaration_index_set* duplicate() const;
      virtual ~C_declaration_index_set();
      virtual void internalExecute(LensContext *);
      virtual void checkChildren();
      virtual void recursivePrint();
      C_index_set* getIndexSet();

   private:
      C_declarator* _declarator;
      C_index_set* _indexSet;
};
#endif
