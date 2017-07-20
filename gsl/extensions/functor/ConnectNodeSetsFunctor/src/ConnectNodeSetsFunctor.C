// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BCM-YKT-11-19-2015
//
// (C) Copyright IBM Corp. 2005-2015  All rights reserved
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

void ConnectNodeSetsFunctor::userInitialize(LensContext* CG_c) 
{
}

void ConnectNodeSetsFunctor::userExecute(LensContext* CG_c, NodeSet*& source, NodeSet*& destination, Functor*& sampling, Functor*& sourceOutAttr, Functor*& destinationInAttr) 
{
   CG_c->connectionContext->reset();
   ConnectionContext* cc = CG_c->connectionContext;

   cc->sourceSet = source;
   cc->destinationSet = destination;

   std::vector<DataItem*> nullArgs;
   std::auto_ptr<DataItem> outAttrRVal;
   std::auto_ptr<DataItem> inAttrRVal;
   std::auto_ptr<DataItem> rval;

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

   while(!cc->done) {
      cc->restart = false;
      cc->currentSample++;

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

      lc->nodeToNode(cc->sourceNode, cc->outAttrPSet, cc->destinationNode, 
		     cc->inAttrPSet, CG_c->sim);

      // call sampfctr2, which will set source and destination nodes 
      // (and maybe other stuff)
      sampling->execute(CG_c, nullArgs, rval);
   }
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

void ConnectNodeSetsFunctor::duplicate(std::auto_ptr<ConnectNodeSetsFunctor>& dup) const
{
   dup.reset(new ConnectNodeSetsFunctor(*this));
}

void ConnectNodeSetsFunctor::duplicate(std::auto_ptr<Functor>& dup) const
{
   dup.reset(new ConnectNodeSetsFunctor(*this));
}

void ConnectNodeSetsFunctor::duplicate(std::auto_ptr<CG_ConnectNodeSetsFunctorBase>& dup) const
{
   dup.reset(new ConnectNodeSetsFunctor(*this));
}

