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

#ifndef _EACHDSTFUNCTOR_H_
#define _EACHDSTFUNCTOR_H_
#include "Copyright.h"

#include "SampFctr2Functor.h"
#include <memory>
#include <list>
#include <vector>
class DataItem;
class LensContext;
class Functor;
class NodeDescriptor;
class NodeSet;

class EachDstFunctor: public SampFctr2Functor
{
   public:
      EachDstFunctor();
      EachDstFunctor(const EachDstFunctor&);
      virtual void duplicate(std::auto_ptr<Functor> &fap) const;
      virtual ~EachDstFunctor();
   protected:
      virtual void doInitialize(LensContext *c, 
				const std::vector<DataItem*>& args);
      virtual void doExecute(LensContext *c, 
			     const std::vector<DataItem*>& args, 
			     std::auto_ptr<DataItem>& rvalue);
   private:
      std::auto_ptr<Functor> _functor_ap;
      bool _isUntouched;
      NodeSet *_destinationSet;
      std::vector<NodeDescriptor*> _nodes;
      std::vector<NodeDescriptor*>::iterator _nodesIter, _nodesEnd;
      int count;

};
#endif
