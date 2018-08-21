// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BCM-YKT-07-18-2017
//
// (C) Copyright IBM Corp. 2005-2017  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#ifndef C_declaration_gridcoord_H
#define C_declaration_gridcoord_H
#include "Copyright.h"

#include "C_declaration.h"

class C_declarator;
class C_gridset;
class LensContext;
class SyntaxError;

class C_declaration_gridcoord : public C_declaration
{
   public:
      C_declaration_gridcoord(const C_declaration_gridcoord&);
      C_declaration_gridcoord(C_declarator *, C_gridset *, SyntaxError *);
      virtual C_declaration_gridcoord* duplicate() const;
      virtual ~C_declaration_gridcoord();
      virtual void internalExecute(LensContext *);
      virtual void checkChildren();
      virtual void recursivePrint();

   private:
      C_declarator* _declarator;
      C_gridset* _gridset;
};
#endif
