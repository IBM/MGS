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

#ifndef C_declaration_int_H
#define C_declaration_int_H
#include "Copyright.h"

#include "C_declaration.h"
#include <memory>
#include <map>

class C_declarator;
class LensContext;
class SyntaxError;

class C_declaration_int : public C_declaration
{
   public:
      C_declaration_int(const C_declaration_int&);
      C_declaration_int(C_declarator *, int, SyntaxError *);
      virtual C_declaration_int* duplicate() const;
      virtual ~C_declaration_int();
      virtual void internalExecute(LensContext *);
      virtual void checkChildren();
      virtual void recursivePrint();

   private:
      C_declarator* _declarator;
      int _intValue;
};
#endif
