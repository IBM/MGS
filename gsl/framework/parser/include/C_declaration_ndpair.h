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

#ifndef C_declaration_ndpair_H
#define C_declaration_ndpair_H
#include "Copyright.h"

#include "C_declaration.h"
#include <memory>
#include <map>

class C_declarator;
class C_ndpair_clause;
class LensContext;
class SyntaxError;

class C_declaration_ndpair : public C_declaration
{
   public:
      C_declaration_ndpair(const C_declaration_ndpair&);
      C_declaration_ndpair(C_declarator *,  C_ndpair_clause *, SyntaxError *);
      virtual C_declaration_ndpair* duplicate() const;
      virtual ~C_declaration_ndpair();
      virtual void internalExecute(LensContext *);
      virtual void checkChildren();
      virtual void recursivePrint();

   private:
      C_declarator* _declarator;
      C_ndpair_clause* _ndp_clause;
};
#endif
