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

#ifndef C_declaration_float_H
#define C_declaration_float_H
#include "Copyright.h"

#include "C_declaration.h"

#include <memory>
#include <map>

class C_constant;
class C_declarator;
class LensContext;
class SyntaxError;

class C_declaration_float : public C_declaration
{
   public:
      C_declaration_float(C_declarator *, C_constant *, SyntaxError *);
      C_declaration_float(const C_declaration_float&);
      virtual C_declaration_float* duplicate() const;
      virtual ~C_declaration_float();
      virtual void internalExecute(LensContext *);
      virtual void checkChildren();
      virtual void recursivePrint();

   private:
      C_constant* _constant;
      C_declarator* _declarator;
      float _floatValue;

};
#endif
