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

#ifndef C_dim_declaration_H
#define C_dim_declaration_H
#include "Copyright.h"

#include "C_production.h"

class C_int_constant_list;
class LensContext;

class C_dim_declaration: public C_production
{
   public:
      C_dim_declaration(const C_dim_declaration&);
      C_dim_declaration(C_int_constant_list *, SyntaxError *);
      virtual ~C_dim_declaration();
      virtual C_dim_declaration* duplicate() const;
      virtual void internalExecute(LensContext *);
      virtual void checkChildren();
      virtual void recursivePrint();
      const C_int_constant_list* getIntConstantList() const;

   private:
      C_int_constant_list* _intConstantList;
};
#endif
