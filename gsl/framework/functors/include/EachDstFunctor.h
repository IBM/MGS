// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef _EACHDSTFUNCTOR_H_
#define _EACHDSTFUNCTOR_H_
#include "Copyright.h"

#include "NumericDataItem.h"
#include "IntDataItem.h"
#include "SampFctr2Functor.h"
#include <memory>
#include <list>
#include <vector>
class DataItem;
class GslContext;
class Functor;
class NodeDescriptor;
class NodeSet;

class EachDstFunctor: public SampFctr2Functor
{
   public:
      EachDstFunctor();
      EachDstFunctor(const EachDstFunctor&);
      virtual void duplicate(std::unique_ptr<Functor>&& fap) const;
      virtual ~EachDstFunctor();
   protected:
      virtual void doInitialize(GslContext *c, 
				const std::vector<DataItem*>& args);
      virtual void doExecute(GslContext *c, 
			     const std::vector<DataItem*>& args, 
			     std::unique_ptr<DataItem>& rvalue);
   private:
      std::unique_ptr<Functor> _functor_ap;
      //bool _isUntouched;
      NodeSet *_destinationSet;
      std::vector<NodeDescriptor*> _nodes;
      std::vector<NodeDescriptor*>::iterator _nodesIter, _nodesEnd;
      int count;
      int _allowConnectToItself;
#define REUSE_MEMORY
#ifdef REUSE_MEMORY
      std::vector<DataItem*> nullArgs;
      //nullArgs.push_back(connectionContext);
      std::unique_ptr<DataItem> rval_ap;
#endif

};
#endif
