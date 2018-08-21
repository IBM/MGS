// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BCM-YKT-07-18-2017
//
// (C) Copyright IBM Corp. 2005-2017  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#ifndef _MANHATTANRING2FUNCTOR_H_
#define _MANHATTANRING2FUNCTOR_H_
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

class ManhattanRing2Functor: public SampFctr1Functor
{
   public:
      ManhattanRing2Functor();
      ManhattanRing2Functor(ManhattanRing2Functor*);
      void collectRadialNodes(
	 std::vector<int> origin, NodeSet* sourceSet, 
	 std::vector<int> & radiusSample,
	 std::vector<std::vector<NodeDescriptor*> >&collectedRadialNodes);
      virtual void duplicate(std::auto_ptr<Functor> &fap) const;
      virtual ~ManhattanRing2Functor();
   protected:
      virtual void doInitialize(LensContext *c, 
				const std::vector<DataItem*>& args);
      virtual void doExecute(LensContext *c, 
			     const std::vector<DataItem*>& args, 
			     std::auto_ptr<DataItem>& rvalue);
   private:
      std::vector<std::vector<NodeDescriptor*> > _sampleSet;
      std::vector<int> _list;
      ConnectionContext::Responsibility _responsibility;
      int _currentSample;
      NodeDescriptor *_refNode;
      unsigned _currentCount;
      unsigned _currentList;
};
#endif
