// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef C_declarator_nodeset_extension_H
#define C_declarator_nodeset_extension_H
#include "Copyright.h"

#include "C_production.h"

class C_declarator;
class C_nodeset_extension;
class LensContext;
class NodeSet;
class SyntaxError;

class C_declarator_nodeset_extension : public C_production
{
   public:
      C_declarator_nodeset_extension(const C_declarator_nodeset_extension&);
      C_declarator_nodeset_extension(C_declarator *, C_nodeset_extension *, 
				     SyntaxError *);
      virtual ~C_declarator_nodeset_extension();
      virtual C_declarator_nodeset_extension* duplicate() const;
      virtual void internalExecute(LensContext *);
      virtual void checkChildren();
      virtual void recursivePrint();
      NodeSet* getNodeSet() {
	 return _nodeset;
      }

   private:
      C_declarator* _declarator;
      C_nodeset_extension* _nodesetExtension;
      NodeSet* _nodeset;
};
#endif
