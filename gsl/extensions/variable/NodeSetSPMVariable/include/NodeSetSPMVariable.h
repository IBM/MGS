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

#ifndef NodeSetSPMVariable_H
#define NodeSetSPMVariable_H
#include "Lens.h"

#include "CG_NodeSetSPMVariable.h"
#include <memory>
#include "ShallowArray.h"

class NodeSetSPMVariable : public CG_NodeSetSPMVariable
{
   public:
      virtual void initialize(RNG&);
      virtual void dca(Trigger* trigger, NDPairList* ndPairList);
      NodeSetSPMVariable();
      virtual ~NodeSetSPMVariable();
      virtual void duplicate(std::auto_ptr<NodeSetSPMVariable>& dup) const;
      virtual void duplicate(std::auto_ptr<Variable>& dup) const;
      virtual void duplicate(std::auto_ptr<CG_NodeSetSPMVariable>& dup) const;

   private:
      unsigned dimx;
      unsigned dimy;
};

#endif
