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

#ifndef C_declaration_service_H
#define C_declaration_service_H
#include "Copyright.h"

#include "C_declaration.h"
#include <memory>
#include <map>

class C_declarator;
class C_service;
class LensContext;
class SyntaxError;

class C_declaration_service : public C_declaration
{
   public:
      C_declaration_service(const C_declaration_service&);
      C_declaration_service(C_declarator *, C_service *, SyntaxError *);
      virtual C_declaration_service* duplicate() const;
      virtual ~C_declaration_service();
      virtual void internalExecute(LensContext *);
      virtual void checkChildren();
      virtual void recursivePrint();

   private:
      C_declarator* _declarator;
      C_service* _service;
};
#endif
