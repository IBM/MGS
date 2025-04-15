// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef C_composite_statement_H
#define C_composite_statement_H
#include "Copyright.h"

#include "C_production.h"

class C_declaration;
class C_directive;
class LensContext;
class SyntaxError;

class C_composite_statement : public C_production
{
   public:
      C_composite_statement(const C_composite_statement&);
      C_composite_statement(C_declaration *, SyntaxError *);
      C_composite_statement(C_directive *, SyntaxError *);
      virtual ~C_composite_statement ();
      virtual C_composite_statement* duplicate() const;
      virtual void internalExecute(LensContext *);
      virtual void checkChildren();
      virtual void recursivePrint();
      C_declaration* getDeclaration() const { 
	 return _declaration; 
      }
      C_directive* getFunctionSpecifier() const { 
	 return _directive; 
      }

   private:
      C_declaration* _declaration;
      C_directive* _directive;
};
#endif
