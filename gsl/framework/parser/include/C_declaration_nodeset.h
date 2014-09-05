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

#ifndef C_declaration_nodeset_H
#define C_declaration_nodeset_H
#include "Copyright.h"

#include "C_declaration.h"
#include <memory>
#include <map>

class C_declarator;
class C_nodeset;
class LensContext;
class SyntaxError;

class C_declaration_nodeset : public C_declaration
{
   public:
      C_declaration_nodeset(const C_declaration_nodeset&);
      C_declaration_nodeset(C_declarator *, C_nodeset *, SyntaxError *);
      virtual C_declaration_nodeset* duplicate() const;
      virtual ~C_declaration_nodeset();
      virtual void internalExecute(LensContext *);
      virtual void checkChildren();
      virtual void recursivePrint();

   private:
      C_declarator* _declarator;
      C_nodeset* _nodeset;
};
#endif
