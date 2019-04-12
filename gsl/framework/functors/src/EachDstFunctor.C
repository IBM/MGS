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
   : //_isUntouched(true), 
   _destinationSet(0), count(0), _allowConnectToItself(1)
{
   _nodesIter = _nodes.begin();
   _nodesEnd = _nodes.end();
}


EachDstFunctor::EachDstFunctor(const EachDstFunctor& csf)
   : //_isUntouched(csf._isUntouched), 
      _destinationSet(csf._destinationSet), 
     _nodes(csf._nodes), count(csf.count), _allowConnectToItself(csf._allowConnectToItself)

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
   int nbrArgs=args.size();
   std::ostringstream baseMsg;
   baseMsg << "\texpected: EachDst(SampFctr1_typefunctor)\n"
	 << "\texpected: EachDst(SampFctr1_typefunctor, int allowConnectToItself )\n"
	 << "\t\tdefault: allowConnectToItself = 1 [acceptable values: 0 or 1] - if the source and the dest node can be the same?" << std::endl;
   if (nbrArgs != 1 and nbrArgs != 2) {
      std::ostringstream msg;
      msg << "Improper number of initialization arguments passed to EachDstFunctor" << std::endl
	 << baseMsg.str();
      throw SyntaxErrorException(msg.str());
   }
   FunctorDataItem* fdi = dynamic_cast<FunctorDataItem*>(args[0]);
   if (fdi == 0) {
	 std::ostringstream msg;
	 msg << "First argument is not a functor" << std::endl
	    << baseMsg.str();
      throw SyntaxErrorException(msg.str());
   }
   if (fdi->getFunctor()) fdi->getFunctor()->duplicate(_functor_ap);
   else {
      throw SyntaxErrorException(
	 "Bad functor argument passed to EachDstFunctor");
   }
   if (nbrArgs==2) {
      NumericDataItem *allowConnectToItselfDI = 
	 dynamic_cast<NumericDataItem*>(args[1]);
      if (allowConnectToItselfDI==0) {
	 std::ostringstream msg;
	 msg << "Second argument is not a number" << std::endl
	    << baseMsg.str();
	 throw SyntaxErrorException(msg.str());
      }
      _allowConnectToItself=unsigned(allowConnectToItselfDI->getInt());
   }
#if ! defined(REUSE_MEMORY)
   std::unique_ptr<IntDataItem> connectItself(new IntDataItem()); 
   connectItself->setInt(_allowConnectToItself); // 1 = allow to connect to itself, 0 = no

#endif
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
      //_isUntouched = false;
      count = 0;
   }
#if defined(SUPPORT_MULTITHREAD_CONNECTION)
   cc->sourceNode = 0;
   cc->destinationNode = 0;
   cc->sourceNodes.resize(0);
   cc->destinationNodes.resize(0);
#endif

   //note: this should be passed from outside, and set default to 0
#if ! defined(REUSE_MEMORY)
   std::unique_ptr<IntDataItem> connectItself(new IntDataItem()); 
   connectItself->setInt(_allowConnectToItself); // 1 = allow to connect to itself, 0 = no

   std::vector<DataItem*> nullArgs;
   //nullArgs.push_back(connectionContext);
   std::unique_ptr<DataItem> rval_ap;
#endif

   cc->destinationNode = cc->sourceRefNode = (*_nodesIter);
   cc->current = ConnectionContext::_SOURCE;
   // about to call another functor, 
   // setting that functor's responsibility to source
#if defined(SUPPORT_MULTITHREAD_CONNECTION)
   if (_nodesIter == _nodesEnd) {
      cc->sourceNode = 0;
      cc->destinationNode = 0;
      cc->sourceNodes.resize(0);
      cc->destinationNodes.resize(0);
      cc->done = true;
   }
   else{
#if defined(USING_SUB_NODESET) || \
      (defined(SUPPORT_MULTITHREAD_CONNECTION) && \
       SUPPORT_MULTITHREAD_CONNECTION == USE_ONLY_MAIN_THREAD)
      //must set restart to true, for approach using sub-nodeset
      cc->restart = true;
#endif
      _functor_ap->execute(c, nullArgs, rval_ap);
      ++_nodesIter;
   }
#else
#if defined(REUSE_NODEACCESSORS)
   if (cc->restart)
   {
   auto _refNode = cc->sourceRefNode;
   c->sim->ND_from_to[c->sim->_currentConnectNodeSet][_refNode] = std::make_pair(std::vector<NodeDescriptor*>(), int());
   //if (c->sim->ND_from_to[c->sim->_currentConnectNodeSet].count(_refNode) == 0)
   //  c->sim->ND_from_to[c->sim->_currentConnectNodeSet][_refNode] = std::make_pair(std::vector<NodeDescriptor*>(), int());
   }
#endif
   _functor_ap->execute(c, nullArgs, rval_ap);
   while(cc->done && _nodesIter!=_nodesEnd) {
      /* cc->done  means completed the SOURCE-nodeset
       *  but there is still node in the DEST-nodeset to investigate
       */
      ++_nodesIter;
      if (_nodesIter == _nodesEnd) {
	 break;
      }
      cc->destinationNode = cc->sourceRefNode = (*_nodesIter);
      cc->current = ConnectionContext::_SOURCE;
      cc->restart = true;
#if defined(REUSE_NODEACCESSORS)
      {
	 auto _refNode = cc->sourceRefNode;
	 c->sim->ND_from_to[c->sim->_currentConnectNodeSet][_refNode] = std::make_pair(std::vector<NodeDescriptor*>(), int());
	 //if (c->sim->ND_from_to[c->sim->_currentConnectNodeSet].count(_refNode) == 0)
	 //  c->sim->ND_from_to[c->sim->_currentConnectNodeSet][_refNode] = std::make_pair(std::vector<NodeDescriptor*>(), int());
      }
#endif
      _functor_ap->execute(c, nullArgs, rval_ap);
      cc->restart = originalRestart;
   }
   if (_nodesIter == _nodesEnd) {
      cc->sourceNode = 0;
      cc->destinationNode = 0;
      cc->done = true;
   }
#endif
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
 (uniform distribution),
 then update the count of generated
 nodes.
* ********************* */
