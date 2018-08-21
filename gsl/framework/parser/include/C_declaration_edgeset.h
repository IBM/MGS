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

#ifndef C_declaration_edgeset_H
#define C_declaration_edgeset_H
#include "Copyright.h"

#include "C_declaration.h"
#include <memory>
#include <map>

class C_declarator;
class C_edgeset;
class LensContext;
class SyntaxError;

class C_declaration_edgeset : public C_declaration
{
   public:
      C_declaration_edgeset(const C_declaration_edgeset&);
      C_declaration_edgeset(C_declarator *, C_edgeset *, SyntaxError * error);
      virtual C_declaration_edgeset* duplicate() const;
      virtual ~C_declaration_edgeset();
      virtual void internalExecute(LensContext *);
      virtual void checkChildren();
      virtual void recursivePrint();

   private:
      C_declarator* _declarator;
      C_edgeset* _edgeset;
};
#endif
