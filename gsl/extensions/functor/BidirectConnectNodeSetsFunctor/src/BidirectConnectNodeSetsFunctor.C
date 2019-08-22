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

#include "BidirectConnectNodeSetsFunctor.h"
#include "CG_BidirectConnectNodeSetsFunctorBase.h"
#include "LensContext.h"
#include "Connector.h"
#include "LensConnector.h"
#include "GranuleConnector.h"
#include "NoConnectConnector.h"
#include "ConnectionContext.h"
#include "SyntaxErrorException.h"
#include "ParameterSetDataItem.h"
/*
#include "NodeDescriptor.h"
#include "ParameterSet.h"
#include "NDPairList.h"
#include "NDPair.h"
#include "UnsignedIntDataItem.h"
*/
#include "Simulation.h"
#include <memory>
#include <map>

void BidirectConnectNodeSetsFunctor::userInitialize(LensContext* CG_c) 
{
}

void BidirectConnectNodeSetsFunctor::userExecute(LensContext* CG_c, NodeSet*& source, NodeSet*& destination, Functor*& sampling, Functor*& sourceOutAttr, Functor*& destinationInAttr, Functor*& destinationOutAttr, Functor*& sourceInAttr) 
{
   CG_c->connectionContext->reset();
   ConnectionContext* cc = CG_c->connectionContext;

   cc->sourceSet = source;
   cc->destinationSet = destination;
   std::vector<DataItem*> nullArgs;
   std::unique_ptr<DataItem> outAttrRVal;
   std::unique_ptr<DataItem> inAttrRVal;
   std::unique_ptr<DataItem> rval;

   // call sampfctr2, which will set source and destination nodes 
   // (and maybe other stuff)
   sampling->execute(CG_c, nullArgs, rval);
   NodeDescriptor* srcNode = cc->sourceNode;
   NodeDescriptor* dstNode = cc->destinationNode;

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
     std::cerr<<"Error, BidirectConnectNodeSetsFunctor : no connection context set!"<<std::endl;
     exit(0);
   }

   //   std::map<NodeDescriptor*, unsigned> indexMap;
   while(!cc->done) {
      cc->restart = false;
      cc->currentSample++;

      cc->sourceNode = dstNode;
      cc->destinationNode = srcNode;

      ParameterSetDataItem *psdi;

      destinationOutAttr->execute(CG_c, nullArgs, outAttrRVal);
      psdi = dynamic_cast<ParameterSetDataItem*>(outAttrRVal.get());
      if (psdi==0) {
         throw SyntaxErrorException(
	    "BidirectConnectNodeSets: OutAttrPSet functor did not return a Parameter Set!");
      }
      cc->outAttrPSet = psdi->getParameterSet();

      sourceInAttr->execute(CG_c, nullArgs, inAttrRVal);
      psdi = dynamic_cast<ParameterSetDataItem*>(inAttrRVal.get());
      if (psdi==0) {
         throw SyntaxErrorException(
	    "BidirectConnectNodeSets: InAttrPSet functor did not return a Parameter Set!");
      }
      cc->inAttrPSet = psdi->getParameterSet();

      lc->nodeToNode(cc->sourceNode, cc->outAttrPSet, cc->destinationNode, 
		     cc->inAttrPSet, CG_c->sim);

      cc->sourceNode = srcNode;
      cc->destinationNode = dstNode;
      /*
      unsigned thisIndex=0;
      if (indexMap.find(cc->sourceNode) == indexMap.end())
	indexMap[cc->sourceNode]=0;
      else thisIndex = ++indexMap[cc->sourceNode];
      */
      sourceOutAttr->execute(CG_c, nullArgs, outAttrRVal);
      psdi = dynamic_cast<ParameterSetDataItem*>(outAttrRVal.get());
      if (psdi==0) {
         throw SyntaxErrorException(
	    "BidirectConnectNodeSets: OutAttrPSet functor did not return a Parameter Set!");
      }
      cc->outAttrPSet = psdi->getParameterSet();

      destinationInAttr->execute(CG_c, nullArgs, inAttrRVal);
      psdi = dynamic_cast<ParameterSetDataItem*>(inAttrRVal.get());
      if (psdi==0) {
         throw SyntaxErrorException(
	    "BidirectConnectNodeSets: InAttrPSet functor did not return a Parameter Set!");
      }
      cc->inAttrPSet = psdi->getParameterSet();
      /*
      NDPairList paramsLocal;
      UnsignedIntDataItem* paramDI = new UnsignedIntDataItem(thisIndex);
      std::unique_ptr<DataItem> paramDI_ap(paramDI);
      NDPair* ndp = new NDPair("index", paramDI_ap);
      paramsLocal.push_back(ndp);
      cc->inAttrPSet->set(paramsLocal);
      */

      lc->nodeToNode(cc->sourceNode, cc->outAttrPSet, cc->destinationNode, 
		     cc->inAttrPSet, CG_c->sim);

      // call sampfctr2, which will set source and destination nodes 
      // (and maybe other stuff)
      
      sampling->execute(CG_c, nullArgs, rval);
      srcNode = cc->sourceNode;
      dstNode = cc->destinationNode;
   }
}

BidirectConnectNodeSetsFunctor::BidirectConnectNodeSetsFunctor() 
   : CG_BidirectConnectNodeSetsFunctorBase()
{
   _noConnector = new NoConnectConnector;
   _granuleConnector = new GranuleConnector;
   _lensConnector = new LensConnector;
}

BidirectConnectNodeSetsFunctor::~BidirectConnectNodeSetsFunctor() 
{
}

void BidirectConnectNodeSetsFunctor::duplicate(std::unique_ptr<BidirectConnectNodeSetsFunctor>& dup) const
{
   dup.reset(new BidirectConnectNodeSetsFunctor(*this));
}

void BidirectConnectNodeSetsFunctor::duplicate(std::unique_ptr<Functor>& dup) const
{
   dup.reset(new BidirectConnectNodeSetsFunctor(*this));
}

void BidirectConnectNodeSetsFunctor::duplicate(std::unique_ptr<CG_BidirectConnectNodeSetsFunctorBase>& dup) const
{
   dup.reset(new BidirectConnectNodeSetsFunctor(*this));
}

