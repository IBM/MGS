// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef C_connection_script_declaration_H
#define C_connection_script_declaration_H
#include "Copyright.h"

#include "C_production.h"

#include <vector>
class LensContext;
class C_declaration;
class C_directive;
class DataItem;
class SyntaxError;

class C_connection_script_declaration : public C_production
{
   public:
      C_connection_script_declaration(
	 const C_connection_script_declaration&);
      C_connection_script_declaration(C_declaration *, SyntaxError *);
      C_connection_script_declaration(bool, C_directive *, SyntaxError *);
      virtual ~C_connection_script_declaration();
      virtual C_connection_script_declaration* duplicate() const;
      virtual void internalExecute(LensContext *);
      virtual void checkChildren();
      virtual void recursivePrint();
      bool isReturn() { 
	 return _return;
      }
      // Will return null if void
      const DataItem* getRVal() const;

   private:
      C_declaration* _declaration;
      C_directive* _directive;
      bool _return;
};
#endif
