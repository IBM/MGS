// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BCM-YKT-11-19-2015
//
// (C) Copyright IBM Corp. 2005-2015  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#ifndef C_declaration_repname_H
#define C_declaration_repname_H
#include "Copyright.h"

#include "C_declaration.h"

class C_declarator;
class C_repname;
class LensContext;

class C_declaration_repname : public C_declaration
{
   public:
      C_declaration_repname(const C_declaration_repname&);
      C_declaration_repname(C_declarator *, C_repname *, SyntaxError *);
      virtual C_declaration_repname* duplicate() const;
      virtual ~C_declaration_repname();
      virtual void internalExecute(LensContext *);
      virtual void checkChildren();
      virtual void recursivePrint();

   private:
      C_declarator *_declarator;
      C_repname *_repname;

};
#endif
