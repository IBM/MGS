// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef C_declaration_publisher_H
#define C_declaration_publisher_H
#include "Copyright.h"

#include "C_declaration.h"
#include <memory>

class C_declarator;
class C_query_path;
class LensContext;
class SyntaxError;

class C_declaration_publisher : public C_declaration
{
   public:
      C_declaration_publisher(const C_declaration_publisher&);
      C_declaration_publisher(C_declarator *, C_query_path *, SyntaxError *);
      virtual C_declaration_publisher* duplicate() const;
      virtual ~C_declaration_publisher();
      virtual void internalExecute(LensContext *);
      virtual void checkChildren();
      virtual void recursivePrint();

   private:
      C_declarator* _declarator;
      C_query_path* _query_path;
};
#endif
