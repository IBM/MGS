// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef _MANHATTANRINGFUNCTOR_H_
#define _MANHATTANRINGFUNCTOR_H_
#include "Copyright.h"

#include "SampFctr1Functor.h"
#include "NodeDescriptor.h"
#include "ConnectionContext.h"
#include <memory>
#include <list>
#include <vector>

class DataItem;
class LensContext;
class NodeSet;

class ManhattanRingFunctor: public SampFctr1Functor
{
   public:
      ManhattanRingFunctor();
      void collectRadialNodes(
	 std::vector<int> origin, NodeSet* sourceSet, 
	 std::vector<int> & radiusSample,
	 std::vector<std::vector<NodeDescriptor*> >&collectedRadialNodes);
      virtual void duplicate(std::unique_ptr<Functor>&& fap) const;
      virtual ~ManhattanRingFunctor();
   protected:
      virtual void doInitialize(LensContext *c, 
				const std::vector<DataItem*>& args);
      virtual void doExecute(LensContext *c, 
			     const std::vector<DataItem*>& args, 
			     std::unique_ptr<DataItem>& rvalue);
   private:
      std::vector<std::vector<NodeDescriptor*> > _sampleSet;
      std::vector<int> _list;
      ConnectionContext::Responsibility _responsibility;
      int _currentSample;
      NodeDescriptor* _refNode;
      unsigned _currentCount;
      unsigned _currentList;
};
#endif
