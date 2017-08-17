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

#ifndef C_declaration_ndpairlist_H
#define C_declaration_ndpairlist_H
#include "Copyright.h"

#include "C_declaration.h"
#include <memory>
#include <map>

class C_declarator;
class C_ndpair_clause_list;
class LensContext;
class SyntaxError;

class C_declaration_ndpairlist : public C_declaration
{
   public:
      C_declaration_ndpairlist(const C_declaration_ndpairlist&);
      C_declaration_ndpairlist(C_declarator *, C_ndpair_clause_list *, 
			       SyntaxError *);
      virtual C_declaration_ndpairlist* duplicate() const;
      virtual ~C_declaration_ndpairlist();
      virtual void internalExecute(LensContext *);
      virtual void checkChildren();
      virtual void recursivePrint();

   private:
      C_declarator* _declarator;
      C_ndpair_clause_list* _ndp_clause_list;
};
#endif
