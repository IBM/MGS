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

#ifndef _EACHSRCFUNCTOR_H_
#define _EACHSRCFUNCTOR_H_
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

class EachSrcFunctor: public SampFctr2Functor
{
   public:
      EachSrcFunctor();
      EachSrcFunctor(const EachSrcFunctor&);
      virtual void duplicate(std::unique_ptr<Functor> &fap) const;
      virtual ~EachSrcFunctor();
   protected:
      virtual void doInitialize(LensContext *c, 
				const std::vector<DataItem*>& args);
      virtual void doExecute(LensContext *c, 
			     const std::vector<DataItem*>& args, 
			     std::unique_ptr<DataItem>& rvalue);
   private:
      std::unique_ptr<Functor> _functor_ap;
      bool _isUntouched;
      NodeSet *_sourceSet;
      std::vector<NodeDescriptor*> _nodes;
      std::vector<NodeDescriptor*>::iterator _nodesIter, _nodesEnd;

};
#endif
