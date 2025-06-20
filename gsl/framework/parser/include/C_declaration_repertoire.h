// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef C_declaration_repertoire_H
#define C_declaration_repertoire_H
#include "Copyright.h"

#include "C_declaration.h"
#include <memory>
#include <map>

class C_repertoire_declaration;
class GslContext;
class SyntaxError;

class C_declaration_repertoire : public C_declaration
{
   public:
      C_declaration_repertoire(const C_declaration_repertoire&);
      C_declaration_repertoire(C_repertoire_declaration *, SyntaxError *);
      virtual C_declaration_repertoire* duplicate() const;
      virtual ~C_declaration_repertoire();
      virtual void internalExecute(GslContext *);
      virtual void checkChildren();
      virtual void recursivePrint();

   private:
      C_repertoire_declaration* _repertoireDeclaration;
};
#endif
