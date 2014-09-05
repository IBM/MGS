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

#include "EachSrcFunctor.h"
#include "LensContext.h"
#include "DataItem.h"
#include "FunctorType.h"
#include "NodeDescriptor.h"
#include "ConnectionContext.h"
#include "FunctorDataItem.h"
#include "NodeSet.h"
#include "InstanceFactoryQueriable.h"
#include "DataItemQueriable.h"
#include "FunctorDataItem.h"
#include "SyntaxErrorException.h"

EachSrcFunctor::EachSrcFunctor()
   : _isUntouched(true), _sourceSet(0)
{
   _nodesIter = _nodes.begin();
   _nodesEnd = _nodes.end();
}


EachSrcFunctor::EachSrcFunctor(const EachSrcFunctor& csf)
   : _isUntouched(csf._isUntouched), _sourceSet(csf._sourceSet), 
     _nodes(csf._nodes)
{
   if (csf._functor_ap.get()) csf._functor_ap->duplicate(_functor_ap);
   _nodesIter = _nodes.begin();
   _nodesEnd = _nodes.end();
}


void EachSrcFunctor::duplicate(std::auto_ptr<Functor> &fap) const
{
   fap.reset(new EachSrcFunctor(*this));
}


EachSrcFunctor::~EachSrcFunctor()
{
}


void EachSrcFunctor::doInitialize(LensContext *c, 
				  const std::vector<DataItem*>& args)
{
   if (args.size() != 1) {
      throw SyntaxErrorException(
	 "Improper number of initialization arguments passed to EachSrcFunctor");
   }
   FunctorDataItem* fdi = dynamic_cast<FunctorDataItem*>(args[0]);
   if (fdi == 0) {
      throw SyntaxErrorException(
	 "Dynamic cast of DataItem to FunctorDataItem failed on EachSrcFunctor");
   }
   if (fdi->getFunctor()) fdi->getFunctor()->duplicate(_functor_ap);
   else {
      throw SyntaxErrorException(
	 "Bad functor argument passed to EachDstFunctor!");
   }
}


void EachSrcFunctor::doExecute(LensContext *c, 
			       const std::vector<DataItem*>& args, 
			       std::auto_ptr<DataItem>& rvalue)
{
   ConnectionContext* cc = c->connectionContext;
   cc->done = false;
   bool originalRestart = cc->restart;
   if (cc->restart) {
      _sourceSet = cc->sourceSet;
      _nodes.clear();
      cc->sourceSet->getNodes(_nodes);
      _nodesIter = _nodes.begin();
      _nodesEnd = _nodes.end();
      _isUntouched = false;
   }

   std::vector<DataItem*> nullArgs;
   std::auto_ptr<DataItem> rval_ap;
   cc->sourceNode = cc->destinationRefNode = (*_nodesIter);
   cc->current = ConnectionContext::_DEST;
   _functor_ap->execute(c, nullArgs, rval_ap);
   while(cc->done && (_nodesIter != _nodesEnd)) {
      ++_nodesIter;
      if (_nodesIter == _nodesEnd) {
	 break;
      }
      cc->sourceNode = cc->destinationRefNode = (*_nodesIter);
      cc->current = ConnectionContext::_DEST;
      cc->restart = true;
      _functor_ap->execute(c, nullArgs, rval_ap);
      cc->restart = originalRestart;
   }
   if (_nodesIter == _nodesEnd) {
      cc->sourceNode = 0;
      cc->destinationNode = 0;
      cc->done = true;
   }
   cc->restart = originalRestart;
}


/* ********************* *

 check whether it's SOURCE or DEST,
 check that the count of generated
 nodes didn't reach limit,
 check that reference node is set
 and then generate a list of nodes
 within the ring and pick one of them,
 otherwise just pick one of them
 (uniform distribuition),
 then update the count of generated
nodes.

* ********************* */
