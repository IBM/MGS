// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
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
class GslContext;
class NodeSet;
class NodeDescriptor;

class RadialSamplerFunctor: public SampFctr1Functor
{
   public:
      RadialSamplerFunctor();
      RadialSamplerFunctor(const RadialSamplerFunctor& rsf);
      virtual void duplicate(std::unique_ptr<Functor>&& fap) const;
      virtual ~RadialSamplerFunctor();
   protected:
      virtual void doInitialize(GslContext *c, 
				const std::vector<DataItem*>& args);
      virtual void doExecute(GslContext *c, 
			     const std::vector<DataItem*>& args, 
			     std::unique_ptr<DataItem>& rvalue);
      float getRelativeProbability(float distance, float scale, 
				   std::vector<float> &histogram);
   private:
      ConnectionContext::Responsibility _responsibility;
      NodeDescriptor *_refNode;
      float _radius;
      int _borderTolerance;
      int _direction;
      std::vector<NodeDescriptor*> _nodes;
      int _currentNode;
      int _nbrNodes;
      std::vector<int> _refcoords;
      double _square_radius;
};
#endif
