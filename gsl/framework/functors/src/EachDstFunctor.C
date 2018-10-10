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

#include "EachDstFunctor.h"
#include "LensContext.h"
#include "DataItem.h"
#include "FunctorType.h"
#include "Node.h"
#include "ConnectionContext.h"
#include "FunctorDataItem.h"
#include "NodeSet.h"
#include "InstanceFactoryQueriable.h"
#include "DataItemQueriable.h"
#include "FunctorDataItem.h"
#include "SyntaxErrorException.h"

EachDstFunctor::EachDstFunctor()
   : _isUntouched(true), _destinationSet(0), count(0)
{
   _nodesIter = _nodes.begin();
   _nodesEnd = _nodes.end();
}


EachDstFunctor::EachDstFunctor(const EachDstFunctor& csf)
   : _isUntouched(csf._isUntouched), _destinationSet(csf._destinationSet), 
     _nodes(csf._nodes), count(csf.count)
{
   if (csf._functor_ap.get()) csf._functor_ap->duplicate(_functor_ap);
   _nodesIter = _nodes.begin();
   _nodesEnd = _nodes.end();
}


void EachDstFunctor::duplicate (std::unique_ptr<Functor> &fap) const
{
   fap.reset(new EachDstFunctor(*this));
}


EachDstFunctor::~EachDstFunctor()
{
}


void EachDstFunctor::doInitialize(LensContext *c, 
				  const std::vector<DataItem*>& args)
{
   if (args.size() != 1) {
      throw SyntaxErrorException(
	 "Improper number of initialization arguments passed to EachDstFunctor");
   }
   FunctorDataItem* fdi = dynamic_cast<FunctorDataItem*>(args[0]);
   if (fdi == 0) {
      throw SyntaxErrorException(
	 "Dynamic cast of DataItem to FunctorDataItem failed on EachDstFunctor");
   }
   if (fdi->getFunctor()) fdi->getFunctor()->duplicate(_functor_ap);
   else {
      throw SyntaxErrorException(
	 "Bad functor argument passed to EachDstFunctor");
   }
}


void EachDstFunctor::doExecute(LensContext *c, 
			       const std::vector<DataItem*>& args, 
			       std::unique_ptr<DataItem>& rvalue)
{
   ConnectionContext* cc = c->connectionContext;
   cc->done = false;
   bool originalRestart = cc->restart;
   if (cc->restart) {
      _destinationSet = cc->destinationSet;
      _nodes.clear();
      _destinationSet->getNodes(_nodes);
      _nodesIter = _nodes.begin();
      _nodesEnd = _nodes.end();
      _isUntouched = false;
      count = 0;
   }

   std::vector<DataItem*> nullArgs;
   std::unique_ptr<DataItem> rval_ap;

   cc->destinationNode = cc->sourceRefNode = (*_nodesIter);
   cc->current = ConnectionContext::_SOURCE;
   // about to call another functor, 
   // setting that functor's responsibility to source
   _functor_ap->execute(c, nullArgs, rval_ap);
   while(cc->done && _nodesIter!=_nodesEnd) {
      ++_nodesIter;
      if (_nodesIter == _nodesEnd) {
	 break;
      }
      cc->destinationNode = cc->sourceRefNode = (*_nodesIter);
      cc->current = ConnectionContext::_SOURCE;
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


/* ***********************
Grab currentSample number
if (first time){
Grab list of destination nodes from the destination nodeset (in a list passed to the NodeSet object)
Set iterator to begin
}

set SourceReferencePoint to Node from iterator
set Destination node from iterator
call SamplingFctr1
if (source node is 0)
{
increment  iterator
if (iterator == end) return 0 as source and destination nodes
}
return
* ************************/

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
