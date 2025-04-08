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

#include "ConnectNodeSetsFunctor.h"
#include "CG_ConnectNodeSetsFunctorBase.h"
#include "LensContext.h"
#include "Connector.h"
#include "LensConnector.h"
#include "GranuleConnector.h"
#include "NoConnectConnector.h"
#include "ConnectionContext.h"
#include "SyntaxErrorException.h"
#include "ParameterSetDataItem.h"
#include "Simulation.h"
#include <memory>

#if defined(SUPPORT_MULTITHREAD_CONNECTION) && SUPPORT_MULTITHREAD_CONNECTION == USE_STATIC_THREADPOOL
#if defined(USE_THREADPOOL_C11)
#include "ThreadPoolC11.h"
#endif
#include <thread>
#endif

void ConnectNodeSetsFunctor::userInitialize(LensContext* CG_c) 
{
}

void ConnectNodeSetsFunctor::userExecute(LensContext* CG_c, NodeSet*& source, NodeSet*& destination, Functor*& sampling, Functor*& sourceOutAttr, Functor*& destinationInAttr) 
{
//#define DEBUG_TIMER
#ifdef DEBUG_TIMER
     if (CG_c->sim->getRank()==0)
     {
       CG_c->sim->benchmark_timelapsed(".. ConnectNodeSetsFunctor (userExecute() start)");
     } 
     CG_c->sim->resetCounter();
#endif
   CG_c->connectionContext->reset();
   ConnectionContext* cc = CG_c->connectionContext;

   cc->sourceSet = source;
   cc->destinationSet = destination;

   std::vector<DataItem*> nullArgs;
   std::unique_ptr<DataItem> outAttrRVal;
   std::unique_ptr<DataItem> inAttrRVal;
   std::unique_ptr<DataItem> rval;

#if defined(SUPPORT_MULTITHREAD_CONNECTION) && SUPPORT_MULTITHREAD_CONNECTION == USE_STATIC_THREADPOOL
#if defined(USE_THREADPOOL_C11)
   {
     CG_c->sim->threadPoolC11->init();
   }
#endif
#endif

   // call sampfctr2, which will set source and destination nodes 
   // (and maybe other stuff)
   sampling->execute(CG_c, nullArgs, rval);

   // loop until one of the nodes is null
   //while(cc->destinationNode!=0 && cc->sourceNode!=0)

   Connector* lc;

   if (CG_c->sim->isGranuleMapperPass()) {
     lc=_noConnector;
   } else if (CG_c->sim->isCostAggregationPass()) {
     lc=_granuleConnector;
   } else if (CG_c->sim->isSimulatePass()) {
     lc=_lensConnector;
   } else {
     std::cerr<<"Error, ConnectNodeSetsFunctor : no connection context set!"<<std::endl;
     exit(0);
   }

#if defined(SUPPORT_MULTITHREAD_CONNECTION)
   bool functor_with_multithread = true;
   if ((cc->sourceNodes.size() == 0 && cc->destinationNodes.size() == 0))
      functor_with_multithread  = false;
   //assert(! (cc->sourceNodes.size() >0 && cc->destinationNodes.size() > 0));
   if (functor_with_multithread)
   {
      while(!cc->done) 
      {
	 cc->restart = false;
	 for (int i = 0; i < cc->destinationNodes.size(); ++i)
	 {
	    cc->currentSample++;
	    cc->destinationNode = cc->destinationNodes[i];

	    ParameterSetDataItem *psdi;
	    sourceOutAttr->execute(CG_c, nullArgs, outAttrRVal);
	    psdi = dynamic_cast<ParameterSetDataItem*>(outAttrRVal.get());
	    if (psdi==0) {
	       throw SyntaxErrorException(
		     "ConnectNodeSets: OutAttrPSet functor did not return a Parameter Set!");
	    }
	    cc->outAttrPSet = psdi->getParameterSet();

	    destinationInAttr->execute(CG_c, nullArgs, inAttrRVal);
	    psdi = dynamic_cast<ParameterSetDataItem*>(inAttrRVal.get());
	    if (psdi==0) {
	       throw SyntaxErrorException(
		     "ConnectNodeSets: InAttrPSet functor did not return a Parameter Set!");
	    }
	    cc->inAttrPSet = psdi->getParameterSet();

#ifdef DEBUG_TIMER
	    CG_c->sim->increaseCounter();
#endif
	    lc->nodeToNode(cc->sourceNode, cc->outAttrPSet, cc->destinationNode, 
		  cc->inAttrPSet, CG_c->sim);
	 }
	 for (int i = 0; i < cc->sourceNodes.size(); ++i)
	 {
	    cc->currentSample++;
	    cc->sourceNode = cc->sourceNodes[i];

	    ParameterSetDataItem *psdi;
	    sourceOutAttr->execute(CG_c, nullArgs, outAttrRVal);
	    psdi = dynamic_cast<ParameterSetDataItem*>(outAttrRVal.get());
	    if (psdi==0) {
	       throw SyntaxErrorException(
		     "ConnectNodeSets: OutAttrPSet functor did not return a Parameter Set!");
	    }
	    cc->outAttrPSet = psdi->getParameterSet();

	    destinationInAttr->execute(CG_c, nullArgs, inAttrRVal);
	    psdi = dynamic_cast<ParameterSetDataItem*>(inAttrRVal.get());
	    if (psdi==0) {
	       throw SyntaxErrorException(
		     "ConnectNodeSets: InAttrPSet functor did not return a Parameter Set!");
	    }
	    cc->inAttrPSet = psdi->getParameterSet();

#ifdef DEBUG_TIMER
	    CG_c->sim->increaseCounter();
#endif
	    lc->nodeToNode(cc->sourceNode, cc->outAttrPSet, cc->destinationNode, 
		  cc->inAttrPSet, CG_c->sim);
	 }

	 // call sampfctr2, which will set source and destination nodes 
	 // (and maybe other stuff)
	 sampling->execute(CG_c, nullArgs, rval);
//#if defined(USE_THREADPOOL_C11)
//2^24  ~ 17million
	 //if (cc->currentSample % 10000000 == 0) 
	 //{
	 //   std::cout << "... passing another batch" << std::endl;
	 //   CG_c->sim->benchmark_timelapsed("..... ");
	 //}
//#endif
      }
   }
   else{
      /* some sampling functor has not implemented multi-threading support */
      while(!cc->done) {
	 cc->restart = false;
	 cc->currentSample++; // i.e. a valid pair of (source,dest) is found
	 // validity is based on SamplingFunctor only
	 // it does not take into account 'Interface' and Predicate of Attributes + UserFunction

	 ParameterSetDataItem *psdi;
	 sourceOutAttr->execute(CG_c, nullArgs, outAttrRVal);
	 psdi = dynamic_cast<ParameterSetDataItem*>(outAttrRVal.get());
	 if (psdi==0) {
	    throw SyntaxErrorException(
		  "ConnectNodeSets: OutAttrPSet functor did not return a Parameter Set!");
	 }
	 cc->outAttrPSet = psdi->getParameterSet();

	 destinationInAttr->execute(CG_c, nullArgs, inAttrRVal);
	 psdi = dynamic_cast<ParameterSetDataItem*>(inAttrRVal.get());
	 if (psdi==0) {
	    throw SyntaxErrorException(
		  "ConnectNodeSets: InAttrPSet functor did not return a Parameter Set!");
	 }
	 cc->inAttrPSet = psdi->getParameterSet();

#ifdef DEBUG_TIMER
	 CG_c->sim->increaseCounter();
#endif
	 lc->nodeToNode(cc->sourceNode, cc->outAttrPSet, cc->destinationNode, 
	       cc->inAttrPSet, CG_c->sim);

	 // call sampfctr2, which will set source and destination nodes 
	 // (and maybe other stuff)
	 sampling->execute(CG_c, nullArgs, rval);
      }
   }
#if defined(USE_THREADPOOL_C11)
   CG_c->sim->threadPoolC11->shutdown();
#endif

#else
   while(!cc->done) {
      cc->restart = false;
      cc->currentSample++; // i.e. a valid pair of (source,dest) is found
       // validity is based on SamplingFunctor only
       // it does not take into account 'Interface' and Predicate of Attributes + UserFunction

      ParameterSetDataItem *psdi;
      sourceOutAttr->execute(CG_c, nullArgs, outAttrRVal);
      psdi = dynamic_cast<ParameterSetDataItem*>(outAttrRVal.get());
      if (psdi==0) {
         throw SyntaxErrorException(
	    "ConnectNodeSets: OutAttrPSet functor did not return a Parameter Set!");
      }
      cc->outAttrPSet = psdi->getParameterSet();

      destinationInAttr->execute(CG_c, nullArgs, inAttrRVal);
      psdi = dynamic_cast<ParameterSetDataItem*>(inAttrRVal.get());
      if (psdi==0) {
         throw SyntaxErrorException(
	    "ConnectNodeSets: InAttrPSet functor did not return a Parameter Set!");
      }
      cc->inAttrPSet = psdi->getParameterSet();

#ifdef DEBUG_TIMER
      CG_c->sim->increaseCounter();
      if (CG_c->sim->getRank() == 0 &&  CG_c->sim->getCounter() % 10000000 == 0)
      {
	 std::cout << ".......... nodeToNode() ... called " << CG_c->sim->getCounter() << " times, and took " /*<< std::endl */;
	 CG_c->sim->benchmark_timelapsed_diff("................................");
      }
#endif
      lc->nodeToNode(cc->sourceNode, cc->outAttrPSet, cc->destinationNode, 
		     cc->inAttrPSet, CG_c->sim);

      // call sampfctr2, which will set source and destination nodes 
      // (and maybe other stuff)
      sampling->execute(CG_c, nullArgs, rval);
   }
#endif
#ifdef DEBUG_TIMER
     if (CG_c->sim->getRank()==0)
     {
	std::string msg;
	if (CG_c->sim->isGranuleMapperPass()) {
	   msg = "_noConnector";
	} else if (CG_c->sim->isCostAggregationPass()) {
	   msg = "_granuleConnector";
	} else if (CG_c->sim->isSimulatePass()) {
	   msg = "_lensConnector";
	} 
	std::cout << ".........." << msg << std::endl;
       CG_c->sim->benchmark_timelapsed(".. ConnectNodeSetsFunctor (userExecute() end)");
	std::cout << ".......... nodeToNode() has been called " << CG_c->sim->getCounter() << " times"<< std::endl;
     } 
#endif
#if defined(REUSE_NODEACCESSORS) and defined(REUSE_EXTRACTED_NODESET_FOR_CONNECTION)
     //must be reset, right before isSimulatePass()
     assert(0); // check again this piece of code
     CG_c->sim->_currentConnectNodeSet++;
     //if (CG_c->sim->isSimulatePass()) {
     //   CG_c->sim->_currentConnectNodeSet++;
     //}
#endif
}

ConnectNodeSetsFunctor::ConnectNodeSetsFunctor() 
   : CG_ConnectNodeSetsFunctorBase()
{
   _noConnector = new NoConnectConnector;
   _granuleConnector = new GranuleConnector;
   _lensConnector = new LensConnector;
}

ConnectNodeSetsFunctor::~ConnectNodeSetsFunctor() 
{
}

void ConnectNodeSetsFunctor::duplicate(std::unique_ptr<ConnectNodeSetsFunctor>&& dup) const
{
   dup.reset(new ConnectNodeSetsFunctor(*this));
}

void ConnectNodeSetsFunctor::duplicate(std::unique_ptr<Functor>&& dup) const
{
   dup.reset(new ConnectNodeSetsFunctor(*this));
}

void ConnectNodeSetsFunctor::duplicate(std::unique_ptr<CG_ConnectNodeSetsFunctorBase>&& dup) const
{
   dup.reset(new ConnectNodeSetsFunctor(*this));
}

