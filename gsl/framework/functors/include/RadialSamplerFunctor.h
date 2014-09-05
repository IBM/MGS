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

#ifndef _RADIALSAMPLERFUNCTOR_H_
#define _RADIALSAMPLERFUNCTOR_H_
#include "Copyright.h"

#include "SampFctr1Functor.h"
#include "Node.h"
#include "ConnectionContext.h"
#include <memory>
#include <list>
#include <vector>

class DataItem;
class LensContext;
class NodeSet;
class NodeDescriptor;

class RadialSamplerFunctor: public SampFctr1Functor
{
   public:
      RadialSamplerFunctor();
      RadialSamplerFunctor(const RadialSamplerFunctor& rsf);
      virtual void duplicate(std::auto_ptr<Functor> &fap) const;
      virtual ~RadialSamplerFunctor();
   protected:
      virtual void doInitialize(LensContext *c, 
				const std::vector<DataItem*>& args);
      virtual void doExecute(LensContext *c, 
			     const std::vector<DataItem*>& args, 
			     std::auto_ptr<DataItem>& rvalue);
      float getRelativeProbability(float distance, float scale, 
				   std::vector<float> &histogram);
   private:
      ConnectionContext::Responsibility _responsibility;
      NodeDescriptor *_refNode;
      float _radius;
      int _borderTolerance;
      std::vector<NodeDescriptor*> _nodes;
      int _currentNode;
      int _nbrNodes;
      std::vector<int> _refcoords;
};
#endif
