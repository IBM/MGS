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

#ifndef C_declaration_H
#define C_declaration_H
#include "Copyright.h"

#include "C_production.h"

class LensContext;

class C_declaration : public C_production
{
   public:
      C_declaration(SyntaxError* error);
      C_declaration(const C_declaration&);
      virtual ~C_declaration();
      virtual C_declaration* duplicate() const = 0;
      virtual void internalExecute(LensContext *);
      virtual void checkChildren();
      virtual void recursivePrint();
};
#endif
