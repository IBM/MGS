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

#ifndef C_relative_nodeset_H
#define C_relative_nodeset_H
#include "Copyright.h"

#include "C_production.h"

class C_gridnodeset;
class C_nodeset_extension;
class LensContext;
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
      virtual void internalExecute(LensContext *);
      virtual void checkChildren();
      virtual void recursivePrint();
      void completeNodeSet(NodeSet* ns);

   private:
      C_gridnodeset *_gridNodeSet;
      C_nodeset_extension *_nodeSetExtension;
      LensContext* _storedContext;
};
#endif
