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
