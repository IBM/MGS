// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BMC-YKT-08-23-2011-2
//
// (C) Copyright IBM Corp. 2005-2014  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

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
