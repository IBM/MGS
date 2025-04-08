// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BMC-YKT-07-18-2017
//
// (C) Copyright IBM Corp. 2005-2017  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#include "Lens.h"
#include "Zipper.h"
#include "CG_ZipperBase.h"
#include "LensContext.h"
#include "Connector.h"
#include "LensConnector.h"
#include "GranuleConnector.h"
#include "NoConnectConnector.h"
#include "ConnectionContext.h"
#include "SyntaxErrorException.h"
#include "ParameterSetDataItem.h"
#include "Simulation.h"
#include "NodeSetDataItem.h"
#include "FunctorDataItem.h"
#include "CustomStringDataItem.h"
#include "IntDataItem.h"
#include "FloatDataItem.h"
#include "NodeSet.h"
#include "NDPairList.h"
#include "NDPair.h"
#include "ParameterSet.h"
#include "RNG.h"
#include "BranchDataStruct.h"
#include "IndexArrayProducer.h"
#include "BranchDataProducer.h"
#include "BranchDataArrayProducer.h"
#include "Node.h"

#include <memory>
#include <cmath>

void Zipper::userInitialize(LensContext* CG_c) 
{
}

void Zipper::userExecute(LensContext* CG_c, std::vector<DataItem*>::const_iterator begin, std::vector<DataItem*>::const_iterator end) 
{
  CG_c->connectionContext->reset();
  ConnectionContext* cc = CG_c->connectionContext;

  std::string mes = "";
  mes = mes + "ZipperFunctor operates on four arguments:\n" +
    "1) source: [Functor (that returns a NodeSet) | NodeSet ]\n" +
    "2) destination: [Functor (that returns a NodeSet) | NodeSet ]\n" +
    "3) Source OutAttrPSet initializer [NDPairList]\n" + 
    "4) Destination InAttrPSet initializer [NDPairList]\n" +
    "5) Name of stored branch proportions: [OPTIONAL String]." ;
  if ((end - begin) != 4 && (end - begin) != 5) {
    throw SyntaxErrorException(mes);
  }

  std::vector<DataItem*>::const_iterator it = begin;
  NodeSetDataItem* sourceNodeSetDI=0;
  NodeSetDataItem* destinationNodeSetDI=0;
  FunctorDataItem* sourceFunctorDI=0;
  FunctorDataItem* destinationFunctorDI=0;

  DataItem* sourceDI=0;
  sourceFunctorDI = dynamic_cast<FunctorDataItem*>(*it);
  std::unique_ptr<DataItem> ap_sourceDI;
  std::vector<NodeDescriptor*> sourceNodes;
  if (sourceFunctorDI) {//passed a functor
    std::vector<DataItem*> nullArgs;
    while (!cc->done) {//keep calling to get all nodes
      sourceFunctorDI->getFunctor()->execute(CG_c, nullArgs, ap_sourceDI);
      sourceDI=ap_sourceDI.get();
      sourceNodeSetDI = dynamic_cast<NodeSetDataItem*>(sourceDI);
      if (sourceNodeSetDI) {
        std::vector<NodeDescriptor*> nodes;
        sourceNodeSetDI->getNodeSet()->getNodes(nodes);
        sourceNodes.insert(sourceNodes.end(), nodes.begin(), nodes.end());
      } 
      else {
        throw SyntaxErrorException("Argument 1 functor did not return a Node Set.\n" + mes);
      }
    }
    cc->done=false;
  }
  else {
    sourceNodeSetDI = dynamic_cast<NodeSetDataItem*>(*it);
    if (sourceNodeSetDI) {
      sourceNodeSetDI->getNodeSet()->getNodes(sourceNodes);
    } 
    else {
      throw SyntaxErrorException("Argument 1 is non-compliant.\n" + mes);
    }
  }

  ++it;

  DataItem* destinationDI=0;
  destinationFunctorDI = dynamic_cast<FunctorDataItem*>(*it);
  std::unique_ptr<DataItem> ap_destinationDI;
  std::vector<NodeDescriptor*> destinationNodes;
  if (destinationFunctorDI) {//passed a functor
    std::vector<DataItem*> nullArgs;
    while (!cc->done) {//keep calling
      destinationFunctorDI->getFunctor()->execute(CG_c, nullArgs, ap_destinationDI);
      destinationDI=ap_destinationDI.get();
      destinationNodeSetDI = dynamic_cast<NodeSetDataItem*>(destinationDI);
      if (destinationNodeSetDI) {
        std::vector<NodeDescriptor*> nodes;
        destinationNodeSetDI->getNodeSet()->getNodes(nodes);
        destinationNodes.insert(destinationNodes.end(), nodes.begin(), nodes.end());
      } 
      else {
        throw SyntaxErrorException("Argument 2 functor did not return a Node Set.\n" + mes);
      }
    }
    cc->done=false;
  }
  else {
    destinationNodeSetDI = dynamic_cast<NodeSetDataItem*>(*it);
    if (destinationNodeSetDI) {
      destinationNodeSetDI->getNodeSet()->getNodes(destinationNodes);
    } 
    else {
      throw SyntaxErrorException("Argument 2 is non-compliant.\n" + mes);
    }
  }

  ++it;

  FunctorDataItem* outAttrFunctorDI=0;
  ParameterSetDataItem* outAttrPSetDI=0;
  outAttrFunctorDI=dynamic_cast<FunctorDataItem*>(*it);    
  if (outAttrFunctorDI==0)
    throw SyntaxErrorException("Argument 3 is non-compliant.\n" + mes);

  ++it;

  FunctorDataItem* inAttrFunctorDI=0;
  ParameterSetDataItem* inAttrPSetDI=0;
  inAttrFunctorDI=dynamic_cast<FunctorDataItem*>(*it);    
  if (inAttrFunctorDI==0)
    throw SyntaxErrorException("Argument 4 is non-compliant.\n" + mes);

  ++it;

  std::string branchPropListName("");
  if (it!=end) {
    StringDataItem* branchPropListNameDI=0;
    branchPropListNameDI = dynamic_cast<StringDataItem*>(*it);    
    if (branchPropListNameDI==0)
      throw SyntaxErrorException("Argument 5 is non-compliant.\n" + mes);
    else
      branchPropListName=branchPropListNameDI->getString();
  }


  int n=0;
  while (n<sourceNodes.size()) {
    if (CG_c->sim->getGranule(*sourceNodes[n])->getPartitionId() != CG_c->sim->getRank())
      sourceNodes.erase(sourceNodes.begin()+n);
    else ++n;
  }

  n=0;
  while (n<destinationNodes.size()) {  
    if (CG_c->sim->getGranule(*destinationNodes[n])->getPartitionId() != CG_c->sim->getRank())
      destinationNodes.erase(destinationNodes.begin()+n);
    else ++n;
  }

  int sz=destinationNodes.size();
  if (sourceNodes.size()!=sz) {
    std::cerr<<"Something is wrong on Zipper. When used properly, source and destination NodeSets contain the same number of nodes!"<<std::endl;
    std::cerr<< "Source: " << sourceNodes.size() << "; while Dest: " << sz << std::endl;
    exit(-1);
  }


  std::vector<double>* branchPropList=0;
  if (branchPropListName!="") branchPropList=&_branchPropListMap[branchPropListName];
  if (branchPropList->size()==0) {
    RNG rng;
    rng.reSeed(irandom(CG_c->sim->getWorkUnitRandomSeedGenerator()), CG_c->sim->getRank());
    for (int n=0; n<sz; ++n) branchPropList->push_back(drandom(rng));
  }
  else if (branchPropList->size()!=sz) {
    std::cerr<<"Error on Zipper! Mismatched probe and branch proportion sizes!"<<std::endl;
    exit(-1);
  }

  std::vector<DataItem*> nullArgs;
  std::unique_ptr<DataItem> outAttrRVal;
  std::unique_ptr<DataItem> inAttrRVal;

  for (int n=0; n<sz; ++n) {//traverse every nodes
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
    cc->sourceNode=sourceNodes[n];
    outAttrFunctorDI->getFunctor()->execute(CG_c, nullArgs, outAttrRVal);
    outAttrPSetDI = dynamic_cast<ParameterSetDataItem*>(outAttrRVal.get());
    if (outAttrPSetDI)
      cc->outAttrPSet = outAttrPSetDI->getParameterSet();
    else
      throw SyntaxErrorException("Zipper: OutAttrPSet functor did not return a Parameter Set!");


    cc->destinationNode=destinationNodes[n];
    inAttrFunctorDI->getFunctor()->execute(CG_c, nullArgs, inAttrRVal);
    inAttrPSetDI = dynamic_cast<ParameterSetDataItem*>(inAttrRVal.get());
    if (inAttrPSetDI) {
      cc->inAttrPSet = inAttrPSetDI->getParameterSet();
      if (branchPropList!=0) {
#if 1
        //James K.
        IntDataItem* idxDI=new IntDataItem(-1);
        std::unique_ptr<DataItem> paramDI_ap(idxDI);
        NDPair* ndpIdx=new NDPair("idx", paramDI_ap);
        FloatDataItem* branchPropDI=new FloatDataItem((*branchPropList)[n]);
        paramDI_ap.reset(branchPropDI);
        NDPair* ndpBranchProp=new NDPair("branchProp", paramDI_ap);
        NDPairList ndpl;
        ndpl.push_back(ndpIdx);
        ndpl.push_back(ndpBranchProp);
        cc->inAttrPSet->set(ndpl);
#else
        //Tuan
        //1. find out the index of compartment and put into 'idx'
        //NOTE: 
        //  1.a. synapse-receptor/cleft use BranchDataArrayProducer and IndexArrayProducer
        //  1.b. compartment/channel use BranchDataProducer
        //start 1.b
        Node* nd;
        BranchDataStruct* 
          bds;
        {
          nd = cc->sourceNode->getNode();
          if (nd)
          {
            bds= 
              (dynamic_cast<BranchDataProducer*>(nd))->CG_get_BranchDataProducer_branchData();
            if (! bds)
            {
              nd = cc->destinationNode->getNode();
              bds= 
                (dynamic_cast<BranchDataProducer*>(nd))->CG_get_BranchDataProducer_branchData();
            }
          }
        }
        //end 1.b
        //start 1.a
        if (0){//IGNORED NOW
          //TUAN ADD PROBE SYNAPSERECEPTOR + CLEFT
          ////BranchDataStruct* []*  branchDataArray;
          nd = cc->sourceNode->getNode();
          ShallowArray< BranchDataStruct* >* 
            bdsArray = 
            (dynamic_cast<BranchDataArrayProducer*>(nd))->CG_get_BranchDataArrayProducer_branchDataArray();
          // int* []* indexArray;
          ShallowArray< int* >* 
            idxArray = 
            (dynamic_cast<IndexArrayProducer*>(nd))->CG_get_IndexArrayProducer_indexArray();
          if (! bdsArray or ! idxArray)
          { 
            nd = cc->sourceNode->getNode();
          }

        }
        //end 1.a

        int cptIdx = std::floor(float(bds->size) * (*branchPropList)[n]);
        IntDataItem* idxDI=new IntDataItem(cptIdx);
        std::unique_ptr<DataItem> paramDI_ap(idxDI);
        NDPair* ndpIdx=new NDPair("idx", paramDI_ap);
        NDPairList ndpl;
        ndpl.push_back(ndpIdx);
        cc->inAttrPSet->set(ndpl);
#endif
      }
    }
    else
      throw SyntaxErrorException("Zipper: InAttrPSet functor did not return a Parameter Set!");

    lc->nodeToNode(cc->sourceNode, cc->outAttrPSet, cc->destinationNode, 
        cc->inAttrPSet, CG_c->sim);
  }
}

Zipper::Zipper() 
   : CG_ZipperBase()
{
   _noConnector = new NoConnectConnector;
   _granuleConnector = new GranuleConnector;
   _lensConnector = new LensConnector;
}

Zipper::Zipper(Zipper const & z) 
  : CG_ZipperBase(), _noConnector(z._noConnector), _granuleConnector(z._granuleConnector), _lensConnector(z._lensConnector), _branchPropListMap(z._branchPropListMap)
{
  //TUAN : why new?
   _noConnector = new NoConnectConnector;
   _granuleConnector = new GranuleConnector;
   _lensConnector = new LensConnector;
}

Zipper::~Zipper() 
{
  if (_noConnector) delete _noConnector;
  if (_granuleConnector) delete _granuleConnector;
  if (_lensConnector) delete _lensConnector;
}

void Zipper::duplicate(std::unique_ptr<Zipper>&& dup) const
{
   dup.reset(new Zipper(*this));
}

void Zipper::duplicate(std::unique_ptr<Functor>&& dup) const
{
   dup.reset(new Zipper(*this));
}

void Zipper::duplicate(std::unique_ptr<CG_ZipperBase>&& dup) const
{
   dup.reset(new Zipper(*this));
}

