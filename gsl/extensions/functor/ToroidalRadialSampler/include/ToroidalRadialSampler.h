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

#ifndef ToroidalRadialSampler_H
#define ToroidalRadialSampler_H
#include "Lens.h"

#include "CG_ToroidalRadialSamplerBase.h"
#include "LensContext.h"
#include "ParameterSet.h"
#include "NodeDescriptor.h"
#include <memory>
#include <vector>

class ToroidalRadialSampler : public CG_ToroidalRadialSamplerBase
{
   public:
      void userInitialize(LensContext* CG_c, float& radius);
      void userExecute(LensContext* CG_c);
      ToroidalRadialSampler();
      virtual ~ToroidalRadialSampler();
      virtual void duplicate(std::auto_ptr<ToroidalRadialSampler>& dup) const;
      virtual void duplicate(std::auto_ptr<Functor>& dup) const;
      virtual void duplicate(std::auto_ptr<CG_ToroidalRadialSamplerBase>& dup) const;

   private:
      NodeDescriptor *_refNode;
      std::vector<NodeDescriptor*> _nodes;
      int _currentNode;
      int _nbrNodes;
      std::vector<int> _refcoords;
      std::vector<int> _nodeSetSize;
};

#endif
