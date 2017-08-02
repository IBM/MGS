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

#ifndef C_declaration_trigger_H
#define C_declaration_trigger_H
#include "Copyright.h"

#include "C_declaration.h"
#include "LensContext.h"
#include <string>
#include <memory>
#include <map>

class C_declarator;
class C_trigger;
class SyntaxError;

class C_declaration_trigger : public C_declaration
{
   public:
      C_declaration_trigger(const C_declaration_trigger&);
      C_declaration_trigger(C_declarator *, C_trigger *, SyntaxError *);
      virtual C_declaration_trigger* duplicate() const;
      virtual ~C_declaration_trigger();
      virtual void internalExecute(LensContext *);
      virtual void checkChildren();
      virtual void recursivePrint();

   private:
      C_declarator* _declarator;
      C_trigger* _trigger;
      std::string* _name;
};
#endif
