// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef _EACHDSTPROPSRCFUNCTOR_H_
#define _EACHDSTPROPSRCFUNCTOR_H_
#include "Copyright.h"

#include "SampFctr2Functor.h"
#include <memory>
#include <list>
#include <vector>
class DataItem;
class GslContext;
class Functor;
class NodeDescriptor;
class NodeSet;

class EachDstPropSrcFunctor: public SampFctr2Functor
{
   public:
      EachDstPropSrcFunctor();
      EachDstPropSrcFunctor(const EachDstPropSrcFunctor&);
      virtual void duplicate(std::unique_ptr<Functor>&& fap) const;
      virtual ~EachDstPropSrcFunctor();
   protected:
      virtual void doInitialize(GslContext *c, 
				const std::vector<DataItem*>& args);
      virtual void doExecute(GslContext *c, 
			     const std::vector<DataItem*>& args, 
			     std::unique_ptr<DataItem>& rvalue);
   private:
      NodeDescriptor* getProportionalNode(GslContext *c);
      std::unique_ptr<Functor> _functor_ap;
      bool _isUntouched;
      NodeSet * _destinationSet;
      NodeSet * _sourceSet;
      std::vector<NodeDescriptor*> _nodes;
      std::vector<NodeDescriptor*>::iterator _nodesIter, _nodesEnd;
      std::vector<float> _slope;
      int _transforms;
      std::vector<int> _coords;
      int _toroidalSpacing;

};
#endif
