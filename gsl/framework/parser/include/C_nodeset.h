// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef _C_NODESET_H_
#define _C_NODESET_H_
#include "Copyright.h"

#include "C_production.h"

class C_gridset;
class C_nodeset_extension;
class C_declarator;
class C_declarator_nodeset_extension;
class LensContext;
class NodeSet;
class SyntaxError;

class C_nodeset : public C_production
{
   public:
      C_nodeset(const C_nodeset&);
      C_nodeset(C_gridset *, SyntaxError *);
      C_nodeset(C_gridset *, C_nodeset_extension *, SyntaxError *);
      C_nodeset(C_declarator *, C_declarator *, SyntaxError *);
      C_nodeset(C_declarator_nodeset_extension *, SyntaxError *);
      virtual ~C_nodeset ();
      virtual C_nodeset* duplicate() const;
      virtual void internalExecute(LensContext *);
      virtual void checkChildren();
      virtual void recursivePrint();
      NodeSet* getNodeSet() {
	 return _nodeset;
      }

   private:
      C_gridset* _gridSet;
      C_nodeset_extension* _nodesetExtension;
      C_declarator* _relNodeSetDecl;
      C_declarator* _gridSetDecl;
      C_declarator_nodeset_extension* _declaratorNodesetExtension;
      NodeSet* _nodeset;

};
#endif
