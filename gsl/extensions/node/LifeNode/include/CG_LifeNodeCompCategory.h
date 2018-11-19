// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM
//
// BCM-YKT-07-18-2017
//
// (C) Copyright IBM Corp. 2005-2017  All rights reserved
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#ifndef CG_LifeNodeCompCategory_H
#define CG_LifeNodeCompCategory_H

#include "Lens.h"
#include "CG_LifeNodeGridLayerData.h"
#include "CG_LifeNodeInAttrPSet.h"
#include "CG_LifeNodeNodeAccessor.h"
#include "CG_LifeNodeOutAttrPSet.h"
#include "CG_LifeNodePSet.h"
#ifdef HAVE_MPI
#include "CG_LifeNodeProxy.h"
#endif
#include "CG_LifeNodeSharedMembers.h"
#include "CG_LifeNodeWorkUnitGridLayers.h"
#include "CG_LifeNodeWorkUnitInstance.h"
#include "CG_LifeNodeWorkUnitShared.h"
#ifdef HAVE_MPI
#include "ConnectionIncrement.h"
#endif
#include "GridLayerData.h"
#include "GridLayerDescriptor.h"
#ifdef HAVE_MPI
#include "IndexedBlockCreator.h"
#endif
#include "LifeNode.h"
#ifdef HAVE_MPI
#include "MemPattern.h"
#endif
#include "NodeCompCategoryBase.h"
#ifdef HAVE_MPI
#include "OutputStream.h"
#endif
#ifdef HAVE_MPI
#include "Phase.h"
#endif
#ifdef HAVE_MPI
#include "ShallowArray.h"
#endif
#include "Simulation.h"
#ifdef HAVE_MPI
#include <cassert>
#endif
#include <deque>
#include <iostream>
#ifdef HAVE_MPI
#include <list>
#endif
#ifdef HAVE_MPI
#include <map>
#endif
#include <memory>
#include <set>
#include <string>

class CG_LifeNode;
class NDPairList;


class CG_LifeNodeCompCategory : public NodeCompCategoryBase
{
   private:

#ifdef HAVE_MPI
      //TUAN TODO
      //// may need to revise this so that it nolonger use 'IndexedBlockCreator'
      // or 
      // // the Index is always 0 [to maintain the original API]
   class CCDemarshaller : public Demarshaller, public IndexedBlockCreator
   {
      friend class CG_LifeNodeCompCategory;
   public:
      typedef std::map<std::string, CG_LifeNodeProxyDemarshaller*> CG_RecvDemarshallers;
      CCDemarshaller(Simulation* sim)
      : Demarshaller()
      {
         _sim = sim;
      }
      int setMemPattern(std::string phaseName, int source, MemPattern* mpptr)
      {
         CG_RecvDemarshallers::iterator diter = CG_recvTemplates.find(phaseName);
         int nBytes=0;
         bool inList = (diter != CG_recvTemplates.end());
         if (inList) {
            inList = inList && (_receiveList.size()!=0);
            if (inList) {
#if defined(HAVE_GPU) 
               auto niter=_receiveList.begin();
#else
               ShallowArray<CG_LifeNodeProxy>::iterator niter=_receiveList.begin();
#endif
               CG_LifeNodeProxyDemarshaller* dm = diter->second;
               std::vector<MPI_Aint> blocs;
               std::vector<int> npls;
               dm->setDestination(&(*niter));
               dm->getBlocks(npls, blocs);
               int npblocks=npls.size();
               assert(npblocks==blocs.size());
               inList = inList && (npblocks!=0);
               if (inList) {
                  std::vector<int> nplengths;
                  std::vector<int> npdispls;
                  MPI_Aint nodeAddress;
                  MPI_Get_address(&(*niter), &nodeAddress);
                  mpptr->orig = reinterpret_cast<char*>(&(*niter));
                  nBytes += npls[0];
                  nplengths.push_back(npls[0]);
                  int dcurr, dprev=blocs[0]-nodeAddress;
                  npdispls.push_back(dprev);
                  for (int i=1; i<npblocks; ++i) {
                     nBytes += npls[i];
                     dcurr = blocs[i]-nodeAddress;;
                     if (dcurr-dprev == npls[i-1])
                        nplengths[nplengths.size()-1] += npls[i];
                     else {
                        npdispls.push_back(dcurr);
                        nplengths.push_back(npls[i]);
                     }
                     dprev=dcurr;
                  }
                  assert(nplengths.size()==npdispls.size());
                  int* pattern = mpptr->allocatePattern(nplengths.size());
                  std::vector<int>::iterator npliter=nplengths.begin(),
                     nplend=nplengths.end(),
                     npditer=npdispls.begin();
                  for (; npliter!=nplend; ++npliter, ++npditer, ++pattern) {
                     *pattern = *npditer;
                     *(++pattern) = *npliter;
                  }
                  int nblocks=_receiveList.size();
                  nBytes *= nblocks;
                  MPI_Aint naddr, prevnaddr;
                  MPI_Get_address(&(*niter), &prevnaddr);
                  int* bdispls = mpptr->allocateOrigDispls(nblocks);
                  bdispls[0]=0;
                  ++niter;
                  for (int i=1; i<nblocks; ++i, ++niter) {
                     MPI_Get_address(&(*niter), &naddr);
                     bdispls[i]=naddr-prevnaddr;
                     prevnaddr=naddr;
                  }
               }
            }
         }
         return nBytes;
      }
      int getIndexedBlock(std::string phaseName, int source, MPI_Datatype* blockType, MPI_Aint& blockLocation)
      {
         CG_RecvDemarshallers::iterator diter = CG_recvTemplates.find(phaseName);
         int nBytes=0;
         bool inList = (diter != CG_recvTemplates.end());
         if (inList) {
            inList = inList && (_receiveList.size()!=0);
            if (inList) {
#if defined(HAVE_GPU) 
               auto niter=_receiveList.begin();
#else

               ShallowArray<CG_LifeNodeProxy>::iterator niter=_receiveList.begin();
#endif
               CG_LifeNodeProxyDemarshaller* dm = diter->second;
               std::vector<MPI_Aint> blocs;
               std::vector<int> npls;
               dm->setDestination(&(*niter));
               dm->getBlocks(npls, blocs);
               int npblocks=npls.size();
               assert(npblocks==blocs.size());
               inList = inList && (npblocks!=0);
               if (inList) {
                  int* nplengths = new int[npblocks];
                  int* npdispls = new int[npblocks];
                  MPI_Aint nodeAddress;
                  MPI_Get_address(&(*niter), &nodeAddress);
                  for (int i=0; i<npblocks; ++i) {
                     nplengths[i]=npls[i];
                     npdispls[i]=blocs[i]-nodeAddress;
                  }
                  MPI_Datatype proxyTypeBasic, proxyType;
                  MPI_Type_indexed(npblocks, nplengths, npdispls, MPI_CHAR, &proxyTypeBasic);
                  MPI_Type_create_resized(proxyTypeBasic, 0, sizeof(CG_LifeNodeProxy), &proxyType);
                  delete [] nplengths;
                  delete [] npdispls;

                  int nblocks=_receiveList.size();
                  int* blengths = new int[nblocks];
                  MPI_Aint* bdispls = new MPI_Aint[nblocks];
                  blockLocation=nodeAddress;
                  for (int i=0; i<nblocks; ++i, ++niter) {
                     npls.clear();
                     blocs.clear();
                     dm->setDestination(&(*niter));
                     dm->getBlocks(npls, blocs);
                     npblocks=npls.size();
                     for (int j=0; j<npblocks; ++j) nBytes += npls[j];
                     blengths[i]=1;
                     MPI_Get_address(&(*niter), &bdispls[i]);
                     bdispls[i]-=blockLocation;
                  }
                  MPI_Type_create_hindexed(nblocks, blengths, bdispls, proxyType, blockType);
                  MPI_Type_free(&proxyType);
                  delete [] blengths;
                  delete [] bdispls;
               }
            }
         }
         return nBytes;
      }
      int demarshall(const char* buffer, int size, bool& rebuildRequested)
      {
         int buffSize = size;

         CG_RecvDemarshallers::iterator diter = CG_recvTemplates.find(_sim->getPhaseName());
         if (diter != CG_recvTemplates.end()) {
            CG_LifeNodeProxyDemarshaller* dm = diter->second;
#if defined(HAVE_GPU) 
            auto &niter = _receiveState;
            auto nend = _receiveList.end();
#else
            ShallowArray<CG_LifeNodeProxy>::iterator &niter = _receiveState;
            ShallowArray<CG_LifeNodeProxy>::iterator nend = _receiveList.end();
#endif
            const char* buff = buffer;
            while (niter!=nend && buffSize!=0) {
               buffSize = dm->demarshall(buff, buffSize, rebuildRequested);
               buff = buffer+(size-buffSize);
               if (buffSize!=0 || dm->done()) {
                  ++niter;
                  if (niter != nend) {
                     dm->setDestination(&(*niter));
                  }
               }
            }
         }
         return buffSize;
      }
      NodeProxyBase* addDestination()
      {
         _receiveList.increaseSizeTo(_receiveList.size()+1);
#if defined(HAVE_GPU) 
    #if PROXY_ALLOCATION == OPTION_3
         int sz = _receiveList.size();
         um_value.increaseSizeTo(sz);
         um_publicValue.increaseSizeTo(sz);
         um_neighbors.increaseSizeTo(sz);
   int MAX_SUBARRAY_SIZE = 20;
   //NOTE: um_neighbors is an array of array
   um_neighbors[sz-1].resize_allocated_subarray(MAX_SUBARRAY_SIZE, Array_Flat<int>::MemLocation::UNIFIED_MEM);
    #endif
#endif
         return &_receiveList[_receiveList.size()-1];
      }
      void reset()
      {
         CG_RecvDemarshallers::iterator diter = CG_recvTemplates.find(_sim->getPhaseName());
         if (diter != CG_recvTemplates.end()) {
            _receiveState= _receiveList.begin();
            CG_recvTemplates[_sim->getPhaseName()]->setDestination(&(*_receiveState));
         }
      }
      bool done()
      {
         CG_RecvDemarshallers::iterator diter = CG_recvTemplates.find(_sim->getPhaseName());
         return (diter == CG_recvTemplates.end() || _receiveState == _receiveList.end());
      }
      virtual ~CCDemarshaller()
      {
      }
#if defined(HAVE_GPU) 
    #if PROXY_ALLOCATION == OPTION_3
   public:
      ShallowArray_Flat<int, Array_Flat<int>::MemLocation::UNIFIED_MEM> um_value;
      ShallowArray_Flat<int, Array_Flat<int>::MemLocation::UNIFIED_MEM> um_publicValue;
      ShallowArray_Flat<ShallowArray_Flat< int*, Array_Flat<int>::MemLocation::UNIFIED_MEM >, Array_Flat<int>::MemLocation::UNIFIED_MEM> um_neighbors;
    #elif PROXY_ALLOCATION == OPTION_4
      int offset;
    #endif
      //ShallowArray<CG_LifeNodeProxy> _receiveList; //On CPU
      //ShallowArray<CG_LifeNodeProxy>::iterator _receiveListIter;
      //ShallowArray<CG_LifeNodeProxy>::iterator _receiveState;

   protected:
      ShallowArray_Flat<CG_LifeNodeProxy> _receiveList; //On CPU, but linear memory
      ShallowArray_Flat<CG_LifeNodeProxy>::iterator _receiveListIter;
      ShallowArray_Flat<CG_LifeNodeProxy>::iterator _receiveState;
      
      //ShallowArray_Flat<CG_LifeNodeProxy, Array_Flat<int>::MemLocation::UNIFIED_MEM> _receiveList; // On UnifiedMem
      //ShallowArray_Flat<CG_LifeNodeProxy>::iterator _receiveListIter;
      //ShallowArray_Flat<CG_LifeNodeProxy>::iterator _receiveState;
#else
      ShallowArray<CG_LifeNodeProxy> _receiveList;
      ShallowArray<CG_LifeNodeProxy>::iterator _receiveListIter;
      ShallowArray<CG_LifeNodeProxy>::iterator _receiveState;
#endif
      CG_RecvDemarshallers CG_recvTemplates;
   private:
      Simulation* _sim;
   };

#endif

   public:
#ifdef HAVE_MPI
      typedef void (CG_LifeNode::*CG_T_SendFunctionPtr)(OutputStream* ) const;
#endif
#ifdef HAVE_MPI
      typedef void (CG_LifeNode::*CG_T_GetSendTypeFunctionPtr)(std::vector<int>&, std::vector<MPI_Aint>&) const;
#endif
      CG_LifeNodeCompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList);
      virtual void getInitializationParameterSet(std::unique_ptr<ParameterSet>& initPSet);
      virtual void getInAttrParameterSet(std::unique_ptr<ParameterSet>& CG_castedPSet);
      virtual void getOutAttrParameterSet(std::unique_ptr<ParameterSet>& CG_castedPSet);
      virtual void CG_InstancePhase_initialize(NodePartitionItem* arg, RNG& rng);
      virtual void CG_InstancePhase_update(NodePartitionItem* arg, RNG& rng);
      virtual void CG_InstancePhase_copy(NodePartitionItem* arg, RNG& rng);
      //#ifdef TEST_USING_GPU_COMPUTING
      void CG_host_initialize(NodePartitionItem* arg, RNG& rng) {} 
      void CG_host_update(NodePartitionItem* arg, RNG& rng) {}
      void CG_host_copy(NodePartitionItem* arg, RNG& rng) {}
      //void __global__ kernel_update(
      //      //RNG& rng
      //      int size
      //      , int tooSparse 
      //      , int tooCrowded
      //      /* check which order is easier to genrate code */
      //      //int tooSparse,  
      //      //int tooCrowded, 
      //      //int size
      //      ); 
      //void __global__ kernel_copy(
      //      //RNG& rng
      //      int size
      //      ); 
      //#endif
      virtual void getWorkUnits();
#ifdef HAVE_MPI
      virtual void addToSendMap(int toPartitionId, Node* node);
#endif
#ifdef HAVE_MPI
      virtual void allocateProxy(int fromPartitionId, NodeDescriptor* nd);
#endif
#ifdef HAVE_MPI
      void addVariableNamesForPhase(std::set<std::string>& namesSet, const std::string& phase);
#endif
#ifdef HAVE_MPI
      virtual void setDistributionTemplates();
#endif
#ifdef HAVE_MPI
      virtual void resetSendProcessIdIterators();
#endif
#ifdef HAVE_MPI
      virtual int getSendNextProcessId();
#endif
#ifdef HAVE_MPI
      virtual bool atSendProcessIdEnd();
#endif
#ifdef HAVE_MPI
      virtual void resetReceiveProcessIdIterators();
#endif
#ifdef HAVE_MPI
      virtual int getReceiveNextProcessId();
#endif
#ifdef HAVE_MPI
      virtual bool atReceiveProcessIdEnd();
#endif
#ifdef HAVE_MPI
      virtual int setMemPattern(std::string phaseName, int dest, MemPattern* mpptr);
#endif
#ifdef HAVE_MPI
      virtual int getIndexedBlock(std::string phaseName, int dest, MPI_Datatype* blockType, MPI_Aint& blockLocation);
#endif
#ifdef HAVE_MPI
      virtual IndexedBlockCreator* getReceiveBlockCreator(int fromPartitionId);
#endif
#ifdef HAVE_MPI
      virtual void send(int pid, OutputStream* os)
      {
         std::map<std::string, CG_T_SendFunctionPtr>::iterator fiter = CG_sendTemplates.find(getSimulation().getPhaseName());
         if (fiter != CG_sendTemplates.end()) {
            CG_T_SendFunctionPtr & function =  (fiter->second);
            ShallowArray<CG_LifeNode*> &nodes = _sendMap[pid];
            ShallowArray<CG_LifeNode*>::iterator nbegin = nodes.begin(), niter, nend = nodes.end();
            for (niter = nbegin; niter!=nend; ++niter) {
               ((*niter)->*(function))(os);
            }
         }
      }
#endif
#ifdef HAVE_MPI
      virtual CG_LifeNodeCompCategory::CCDemarshaller* getDemarshaller(int fromPartitionId);
#endif
#ifdef HAVE_MPI
      virtual CG_LifeNodeCompCategory::CCDemarshaller* findDemarshaller(int fromPartitionId);
#endif
      CG_LifeNodeSharedMembers& getSharedMembers();
      virtual void getNodeAccessor(std::unique_ptr<NodeAccessor>& nodeAccessor, GridLayerDescriptor* gridLayerDescriptor);
      void allocateNode(NodeDescriptor* nd);
      void allocateNodes(size_t size);
      void allocateProxies(const std::vector<int>& sizes);
      int getNbrComputationalUnits();
      ConnectionIncrement* getComputeCost();
      virtual ~CG_LifeNodeCompCategory();
      static CG_LifeNodeSharedMembers* CG_sharedMembers;
#if defined(HAVE_GPU) 
      std::map <int, CCDemarshaller*>& getDemarshallerMap(){ return _demarshallerMap; };
#endif
   protected:
      //TUAN TODO
      //we need to revise data in _sendMap and ...
#ifdef HAVE_MPI
      std::map <int, CCDemarshaller*> _demarshallerMap;
#endif
#ifdef HAVE_MPI
      std::map <int, CCDemarshaller*>::iterator _demarshallerMapIter;
#endif
#ifdef HAVE_MPI
      std::map <int, ShallowArray<CG_LifeNode*> > _sendMap;
#endif
#ifdef HAVE_MPI
      std::map <int, ShallowArray<CG_LifeNode*> >::iterator _sendMapIter;
#endif
#if defined(HAVE_GPU) 
   public:
      //TUAN: we can use 'public' or derive a function (with auto-generated name)
    #if PROXY_ALLOCATION == OPTION_4
      ShallowArray_Flat<int, Array_Flat<int>::MemLocation::UNIFIED_MEM> proxy_um_value;
      ShallowArray_Flat<int, Array_Flat<int>::MemLocation::UNIFIED_MEM> proxy_um_publicValue;
      ShallowArray_Flat<ShallowArray_Flat< int*, Array_Flat<int>::MemLocation::UNIFIED_MEM >, 
         Array_Flat<int>::MemLocation::UNIFIED_MEM> proxy_um_neighbors;
    #endif

      ShallowArray_Flat<int, Array_Flat<int>::MemLocation::UNIFIED_MEM> um_value;
      //TUAN TODO test to see if we can reduce 'InitNode' execution time
   #ifdef TEST_INITNODE_ON_CPU
      ShallowArray_Flat<int, Array_Flat<int>::MemLocation::CPU> init_um_value; //for storing data from 'InitNode' statement
   #endif
      ShallowArray_Flat<int, Array_Flat<int>::MemLocation::UNIFIED_MEM> um_publicValue;
   #if DATAMEMBER_ARRAY_ALLOCATION == OPTION_3
      ShallowArray_Flat<ShallowArray_Flat< int*, Array_Flat<int>::MemLocation::UNIFIED_MEM >, 
         Array_Flat<int>::MemLocation::UNIFIED_MEM> um_neighbors;
      //int* um_value; //to be allocated using cudaMallocManaged
      //int* um_publicValue; //to be allocated using cudaMallocManaged
      //std::vector<int*> um_neighbors;
      //ShallowArray_Flat<ShallowArray< int*, Array_Flat<int>::MemLocation::UNIFIED_MEM >, Array_Flat<int>::MemLocation::UNIFIED_MEM> um_neighbors;
   #elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_4
      ShallowArray_Flat<int*, Array_Flat<int>::MemLocation::UNIFIED_MEM > um_neighbors;
      //always 'int' for the two below arrays
      ShallowArray_Flat<int, Array_Flat<int>::MemLocation::UNIFIED_MEM > um_neighbors_start_offset;
      ShallowArray_Flat<int, Array_Flat<int>::MemLocation::UNIFIED_MEM > um_neighbors_num_elements;
   #elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_4b
      ShallowArray_Flat<int*, Array_Flat<int>::MemLocation::UNIFIED_MEM > um_neighbors;
      //always 'int' for the two below arrays
      ShallowArray_Flat<int, Array_Flat<int>::MemLocation::UNIFIED_MEM > um_neighbors_num_elements;
      int um_neighbors_max_elements;
   #elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_5
      ShallowArray_Flat<ShallowArray_Flat< int*, Array_Flat<int>::MemLocation::UNIFIED_MEM > um_neighbors;
      //always 'int' for the below array
      ShallowArray_Flat<int, Array_Flat<int>::MemLocation::UNIFIED_MEM > um_neighbors_start_offset;
   #endif
   protected:
      ShallowArray_Flat<LifeNode, Array_Flat<int>::MemLocation::CPU, 1000> _nodes;
#else
      ShallowArray<LifeNode, 1000, 4> _nodes;
#endif
      ConnectionIncrement _computeCost;
   private:
#ifdef HAVE_MPI
      std::map<std::string, CG_T_SendFunctionPtr> CG_sendTemplates;
#endif
#ifdef HAVE_MPI
      std::map<std::string, CG_T_GetSendTypeFunctionPtr> CG_getSendTypeTemplates;
#endif
};

#endif
