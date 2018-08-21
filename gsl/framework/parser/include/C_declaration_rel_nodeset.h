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

#ifndef C_declaration_rel_nodeset_H
#define C_declaration_rel_nodeset_H
#include "Copyright.h"

#include "C_declaration.h"
class C_declarator;
class C_relative_nodeset;
class LensContext;
class SyntaxError;

class C_declaration_rel_nodeset : public C_declaration
{
   public:
      C_declaration_rel_nodeset(const C_declaration_rel_nodeset&);
      C_declaration_rel_nodeset(C_declarator *, C_relative_nodeset *, 
				SyntaxError *);
      virtual C_declaration_rel_nodeset* duplicate() const;
      virtual ~C_declaration_rel_nodeset();
      virtual void internalExecute(LensContext *);
      virtual void checkChildren();
      virtual void recursivePrint();

   private:
      C_declarator* _declaration;
      C_relative_nodeset* _relativeNodeSet;

};
#endif
