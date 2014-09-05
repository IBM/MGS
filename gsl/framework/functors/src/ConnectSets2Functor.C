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

#include "ConnectSets2Functor.h"
#include "LensContext.h"
#include "ConnectionContext.h"
#include "DataItem.h"
#include "NodeSet.h"
#include "Grid.h"
#include "Repertoire.h"
#include "NodeSetDataItem.h"
#include "EdgeTypeDataItem.h"
#include "FunctorDataItem.h"
#include "IntDataItem.h"
#include "FunctorType.h"
#include "Connector.h"
#include "NoConnectConnector.h"
#include "GranuleConnector.h"
#include "LensConnector.h"
#include "ParameterSetDataItem.h"
#include "InstanceFactoryQueriable.h"
#include "DataItemQueriable.h"
#include "FunctorDataItem.h"
#include "SyntaxErrorException.h"
#include "Simulation.h"

ConnectSets2Functor::ConnectSets2Functor()
{
   _noConnector = new NoConnectConnector;
   _granuleConnector = new GranuleConnector;
   _lensConnector = new LensConnector;
}

void ConnectSets2Functor::duplicate (std::auto_ptr<Functor> &fap) const
{
   fap.reset(new ConnectSets2Functor(*this));
}


ConnectSets2Functor::~ConnectSets2Functor()
{
}


void ConnectSets2Functor::doInitialize(LensContext *c, 
				       const std::vector<DataItem*>& args)
{

}


/*
  Set all values of ConnectionContext to 0
  Set responsibility to _BOTH
  Grab arguments and set NodeSets and EdgeType
  call sampfctr2, which will set source and destination 
  nodes (and maybe other stuff)
  if either of these nodes are null the functor returns
  Otherwise,
  Go to edge init functor to get parameter set
  go to inattr and outattr functors to get parameter sets
  call connect on LensConnector

  The prototype is ConnectSets2(NodeSet from, NodeSet to, 
  EdgeType e, SamplingFctr2 sf,
  Functor einit, Functor outAttr, Functor inAttr);
*/

void ConnectSets2Functor::doExecute(LensContext *c, 
				    const std::vector<DataItem*>& args, 
				    std::auto_ptr<DataItem>& rvalue)
{
   c->connectionContext->reset();
   ConnectionContext* cc = c->connectionContext;

   // Grab arguments 

   if (args.size()!=7) {
      std::string mes = "ConnectSets2: invalid arguments\n";
      mes += "\texpected: ConnectSets2(NodeSet from, NodeSet to, EdgeType e, ";
      mes += "\tSamplingFctr2 sf, int numberSamples, ";
      mes += "\tFunctor einit, Functor outAttr, Functor inAttr)";
      throw SyntaxErrorException(mes);
   }
   NodeSetDataItem *fromDI = dynamic_cast<NodeSetDataItem*>(args[0]);
   if (fromDI==0) {
      throw SyntaxErrorException(
	 "ConnectSets2: argument 1 is not a NodeSetDataItem");
   }
   cc->sourceSet = fromDI->getNodeSet();

   NodeSetDataItem *toDI = dynamic_cast<NodeSetDataItem*>(args[1]);
   if (toDI==0) {
      throw SyntaxErrorException(
	 "ConnectSets2: argument 2 is not a NodeSetDataItem");
   }
   cc->destinationSet = toDI->getNodeSet();

   EdgeTypeDataItem *etDI = dynamic_cast<EdgeTypeDataItem*>(args[2]);
   if (etDI==0) {
      throw SyntaxErrorException(
	 "ConnectSets2: argument 3 is not a EdgeTypeDataItem");
   }
   cc->edgeType = etDI->getEdgeType();

   FunctorDataItem *sfDI = dynamic_cast<FunctorDataItem*>(args[3]);
   if (sfDI==0) {
      throw SyntaxErrorException(
	 "ConnectSets2: argument 4 is not a SamplingFctr2");
   }
   Functor *sf = sfDI->getFunctor();

   FunctorDataItem *einitDI = dynamic_cast<FunctorDataItem*>(args[4]);
   if (einitDI==0) {
      throw SyntaxErrorException("ConnectSets2: argument 5 is not a Functor");
   }
   Functor *einit = einitDI->getFunctor();

   FunctorDataItem *outAttrDI = dynamic_cast<FunctorDataItem*>(args[5]);
   if (outAttrDI==0) {
      throw SyntaxErrorException("ConnectSets2: argument 6 is not a Functor");
   }
   Functor *outAttr = outAttrDI->getFunctor();

   FunctorDataItem *inAttrDI = dynamic_cast<FunctorDataItem*>(args[6]);
   if (inAttrDI==0) {
      throw SyntaxErrorException("ConnectSets2: argument 7 is not a Functor");
   }
   Functor *inAttr = inAttrDI->getFunctor();

   std::vector<DataItem*> nullArgs;
   std::auto_ptr<DataItem> einitRVal;
   std::auto_ptr<DataItem> outAttrRVal;
   std::auto_ptr<DataItem> inAttrRVal;
   std::auto_ptr<DataItem> rval;

   // call sampfctr2, which will set source and destination 
   // nodes (and maybe other stuff)
   sf->execute(c, nullArgs, rval);

   Connector* lc;

   if (c->sim->isGranuleMapperPass()) {
     lc=_noConnector;
   } else if (c->sim->isCostAggregationPass()) {
     lc=_granuleConnector;
   } else if (c->sim->isSimulatePass()) {
     lc=_lensConnector;
   } else {
     std::cerr<<"Error, ConnectSets2Functor : no connection context set!"<<std::endl;
     exit(0);
   }

   while(!cc->done) {
      cc->restart = false;
      cc->currentSample++;
      einit->execute(c, nullArgs, einitRVal);
      ParameterSetDataItem *psdi = dynamic_cast<ParameterSetDataItem*>(einitRVal.get());
      if (psdi==0) {
         throw SyntaxErrorException(
	    "ConnectSets2: EdgeInitializer functor did not return a Parameter Set!");
      }
      cc->edgeInitPSet = psdi->getParameterSet();

      outAttr->execute(c, nullArgs, outAttrRVal);
      psdi = dynamic_cast<ParameterSetDataItem*>(outAttrRVal.get());
      if (psdi==0) {
         throw SyntaxErrorException(
	    "ConnectSets2: OutAttrPSet functor did not return a Parameter Set!");
      }
      cc->outAttrPSet = psdi->getParameterSet();

      inAttr->execute(c, nullArgs, inAttrRVal);
      psdi = dynamic_cast<ParameterSetDataItem*>(inAttrRVal.get());
      if (psdi==0) {
         throw SyntaxErrorException("ConnectSets2: InAttrPSet functor did not return a Parameter Set!");
      }
      cc->inAttrPSet = psdi->getParameterSet();

      lc->nodeToNodeWithEdge(cc->edgeType,cc->edgeInitPSet, cc->sourceNode, 
			     cc->outAttrPSet, cc->destinationNode, 
			     cc->inAttrPSet, c->sim);

      // call sampfctr2, which will set source and destination 
      // nodes (and maybe other stuff)
      sf->execute(c, nullArgs, rval);
   }
}
