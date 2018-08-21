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

#ifndef C_declaration_typedef_H
#define C_declaration_typedef_H
#include "Copyright.h"

#include "C_declaration.h"

class C_typedef_declaration;
class LensContext;
class SyntaxError;

class C_declaration_typedef : public C_declaration
{
   public:
      C_declaration_typedef(const C_declaration_typedef&);
      C_declaration_typedef(C_typedef_declaration *, SyntaxError *);
      virtual C_declaration_typedef* duplicate() const;
      virtual ~C_declaration_typedef();
      virtual void internalExecute(LensContext *);
      virtual void checkChildren();
      virtual void recursivePrint();

   private:
      C_typedef_declaration* _t;
};
#endif
