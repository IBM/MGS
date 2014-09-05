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
