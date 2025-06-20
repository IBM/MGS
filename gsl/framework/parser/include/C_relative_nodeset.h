// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef C_relative_nodeset_H
#define C_relative_nodeset_H
#include "Copyright.h"

#include "C_production.h"

class C_gridnodeset;
class C_nodeset_extension;
class GslContext;
class NodeSet;
class SyntaxError;

class C_relative_nodeset : public C_production
{
   public:
      C_relative_nodeset(const C_relative_nodeset&);
      C_relative_nodeset(C_gridnodeset *, SyntaxError *);
      C_relative_nodeset(C_gridnodeset *, C_nodeset_extension *, 
			 SyntaxError *);
      virtual ~C_relative_nodeset();
      virtual C_relative_nodeset* duplicate() const;
      virtual void internalExecute(GslContext *);
      virtual void checkChildren();
      virtual void recursivePrint();
      void completeNodeSet(NodeSet* ns);

   private:
      C_gridnodeset *_gridNodeSet;
      C_nodeset_extension *_nodeSetExtension;
      GslContext* _storedContext;
};
#endif
