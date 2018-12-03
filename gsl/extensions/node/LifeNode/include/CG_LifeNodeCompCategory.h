// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM
//
// BCM-YKT-12-03-2018
//
//  (C) Copyright IBM Corp. 2005-2018  All rights reserved   .
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
#if defined(HAVE_MPI)
#include "CG_LifeNodeProxy.h"
#endif
#include "CG_LifeNodeSharedMembers.h"
#include "CG_LifeNodeWorkUnitGridLayers.h"
#include "CG_LifeNodeWorkUnitInstance.h"
#include "CG_LifeNodeWorkUnitShared.h"
#if defined(HAVE_MPI)
#include "ConnectionIncrement.h"
#endif
#include "GridLayerData.h"
#include "GridLayerDescriptor.h"
#if defined(HAVE_MPI)
#include "IndexedBlockCreator.h"
#endif
#include "LifeNode.h"
#if defined(HAVE_MPI)
#include "MemPattern.h"
#endif
#include "NodeCompCategoryBase.h"
#if defined(HAVE_MPI)
#include "OutputStream.h"
#endif
#if defined(HAVE_MPI)
#include "Phase.h"
#endif
#if defined(HAVE_MPI)
#include "ShallowArray.h"
#endif
#include "Simulation.h"
#if defined(HAVE_MPI)
#include <cassert>
#endif
#include <deque>
#include <iostream>
#if defined(HAVE_MPI)
#include <list>
#endif
#if defined(HAVE_MPI)
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

#if defined(HAVE_MPI)
   class CCDemarshaller : public Demarshaller, public IndexedBlockCreator
   {
      friend class CG_LifeNodeCompCategory;
   public:
      typedef std::map<std::string, CG_LifeNodeProxyDemarshaller*> CG_RecvDemarshallers;
      CCDemarshaller(Simulation* sim)
      : Demarshaller()      {
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
               auto niter=_receiveList.begin();
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
               auto niter=_receiveList.begin();
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
            auto  &niter = _receiveState;
            auto nend = _receiveList.end();
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
         int MAX_SUBARRAY_SIZE = 20;
         um_value.increaseSizeTo(sz);
         um_publicValue.increaseSizeTo(sz);
         um_neighbors.increaseSizeTo(sz);
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
      ShallowArray_Flat<int,Array_Flat<int>::MemLocation::UnifiedMem> um_value;
#endif
#endif
#if defined(HAVE_GPU)
#if PROXY_ALLOCATION == OPTION_3
      ShallowArray_Flat<int,Array_Flat<int>::MemLocation::UnifiedMem> um_publicValue;
#endif
#endif
#if defined(HAVE_GPU)
#if PROXY_ALLOCATION == OPTION_3
      ShallowArray_Flat<ShallowArray_Flat<int*, Array_Flat<int>::MemLocation::UnifiedMem>,Array_Flat<int>::MemLocation::UnifiedMem> um_neighbors;
#endif
#endif
#if defined(HAVE_GPU)
#if PROXY_ALLOCATION == OPTION_4
      int offset;
#endif
#endif
   protected:
#if ! defined(HAVE_GPU)
      ShallowArray<CG_LifeNodeProxy> _receiveList;
#endif
#if defined(HAVE_GPU)
      ShallowArray_Flat<CG_LifeNodeProxy> _receiveList;
#endif
#if ! defined(HAVE_GPU)
      ShallowArray<CG_LifeNodeProxy>::iterator _receiveListIter;
#endif
#if defined(HAVE_GPU)
      ShallowArray_Flat<CG_LifeNodeProxy>::iterator _receiveListIter;
#endif
#if ! defined(HAVE_GPU)
      ShallowArray<CG_LifeNodeProxy>::iterator _receiveState;
#endif
#if defined(HAVE_GPU)
      ShallowArray_Flat<CG_LifeNodeProxy>::iterator _receiveState;
#endif
      CG_RecvDemarshallers CG_recvTemplates;
   private:
      Simulation* _sim;
   };

#endif

   public:
#if defined(HAVE_MPI)
      typedef void (CG_LifeNode::*CG_T_SendFunctionPtr)(OutputStream* ) const;
#endif
#if defined(HAVE_MPI)
      typedef void (CG_LifeNode::*CG_T_GetSendTypeFunctionPtr)(std::vector<int>&, std::vector<MPI_Aint>&) const;
#endif
      CG_LifeNodeCompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList);
      virtual void getInitializationParameterSet(std::unique_ptr<ParameterSet>& initPSet);
      virtual void getInAttrParameterSet(std::unique_ptr<ParameterSet>& CG_castedPSet);
      virtual void getOutAttrParameterSet(std::unique_ptr<ParameterSet>& CG_castedPSet);
      virtual void CG_InstancePhase_initialize(NodePartitionItem* arg, CG_LifeNodeWorkUnitInstance* wu);
      virtual void CG_InstancePhase_update(NodePartitionItem* arg, CG_LifeNodeWorkUnitInstance* wu);
      virtual void CG_InstancePhase_copy(NodePartitionItem* arg, CG_LifeNodeWorkUnitInstance* wu);
      virtual void getWorkUnits();
#if defined(HAVE_MPI)
      virtual void addToSendMap(int toPartitionId, Node* node);
#endif
#if defined(HAVE_MPI)
      virtual void allocateProxy(int fromPartitionId, NodeDescriptor* nd);
#endif
#if defined(HAVE_MPI)
      void addVariableNamesForPhase(std::set<std::string>& namesSet, const std::string& phase);
#endif
#if defined(HAVE_MPI)
      virtual void setDistributionTemplates();
#endif
#if defined(HAVE_MPI)
      virtual void resetSendProcessIdIterators();
#endif
#if defined(HAVE_MPI)
      virtual int getSendNextProcessId();
#endif
#if defined(HAVE_MPI)
      virtual bool atSendProcessIdEnd();
#endif
#if defined(HAVE_MPI)
      virtual void resetReceiveProcessIdIterators();
#endif
#if defined(HAVE_MPI)
      virtual int getReceiveNextProcessId();
#endif
#if defined(HAVE_MPI)
      virtual bool atReceiveProcessIdEnd();
#endif
#if defined(HAVE_MPI)
      virtual int setMemPattern(std::string phaseName, int dest, MemPattern* mpptr);
#endif
#if defined(HAVE_MPI)
      virtual int getIndexedBlock(std::string phaseName, int dest, MPI_Datatype* blockType, MPI_Aint& blockLocation);
#endif
#if defined(HAVE_MPI)
      virtual IndexedBlockCreator* getReceiveBlockCreator(int fromPartitionId);
#endif
#if defined(HAVE_MPI)
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
#if defined(HAVE_MPI)
      virtual CG_LifeNodeCompCategory::CCDemarshaller* getDemarshaller(int fromPartitionId);
#endif
#if defined(HAVE_MPI)
      virtual CG_LifeNodeCompCategory::CCDemarshaller* findDemarshaller(int fromPartitionId);
#endif
      CG_LifeNodeSharedMembers& getSharedMembers();
      virtual void getNodeAccessor(std::unique_ptr<NodeAccessor>& nodeAccessor, GridLayerDescriptor* gridLayerDescriptor);
      void allocateNode(NodeDescriptor* nd);
      void allocateNodes(size_t size);
      void allocateProxies(const std::vector<size_t>& sizes);
      int getNbrComputationalUnits();
      ConnectionIncrement* getComputeCost();
      virtual ~CG_LifeNodeCompCategory();
      virtual void CG_host_initialize(NodePartitionItem* arg, CG_LifeNodeWorkUnitInstance* wu);
      virtual void CG_host_update(NodePartitionItem* arg, CG_LifeNodeWorkUnitInstance* wu);
      virtual void CG_host_copy(NodePartitionItem* arg, CG_LifeNodeWorkUnitInstance* wu);
#if defined(HAVE_GPU)
#if PROXY_ALLOCATION == OPTION_4
      ShallowArray_Flat<int, Array_Flat<int>::MemLocation::UNIFIED_MEM> proxy_um_value;
#endif
#endif
#if defined(HAVE_GPU)
      ShallowArray_Flat<int, Array_Flat<int>::MemLocation::UNIFIED_MEM> um_value;
#endif
#if defined(HAVE_GPU)
#if PROXY_ALLOCATION == OPTION_4
      ShallowArray_Flat<int, Array_Flat<int>::MemLocation::UNIFIED_MEM> proxy_um_publicValue;
#endif
#endif
#if defined(HAVE_GPU)
      ShallowArray_Flat<int, Array_Flat<int>::MemLocation::UNIFIED_MEM> um_publicValue;
#endif
#if defined(HAVE_GPU)
#if PROXY_ALLOCATION == OPTION_4
      ShallowArray_Flat<ShallowArray_Flat< int* , Array_Flat<int>::MemLocation::UNIFIED_MEM>, Array_Flat<int>::MemLocation::UNIFIED_MEM> proxy_um_neighbors;
#endif
#endif
#if defined(HAVE_GPU)
#if DATAMEMBER_ARRAY_ALLOCATION == OPTION_3
      ShallowArray_Flat<ShallowArray_Flat< int* , Array_Flat<int>::MemLocation::UNIFIED_MEM>, Array_Flat<int>::MemLocation::UNIFIED_MEM> um_neighbors;
#endif
#endif
#if defined(HAVE_GPU)
#if DATAMEMBER_ARRAY_ALLOCATION == OPTION_4
      ShallowArray_Flat<< int* , Array_Flat<int>::MemLocation::UNIFIED_MEM> um_neighbors;
#endif
#endif
#if defined(HAVE_GPU)
#if DATAMEMBER_ARRAY_ALLOCATION == OPTION_4
      ShallowArray_Flat<< int* , Array_Flat<int>::MemLocation::UNIFIED_MEM> um_neighbors_start_offset;
#endif
#endif
#if defined(HAVE_GPU)
#if DATAMEMBER_ARRAY_ALLOCATION == OPTION_4
      ShallowArray_Flat<< int* , Array_Flat<int>::MemLocation::UNIFIED_MEM> um_neighbors_num_elements;
#endif
#endif
#if defined(HAVE_GPU)
#if DATAMEMBER_ARRAY_ALLOCATION == OPTION_4b
      ShallowArray_Flat<< int* , Array_Flat<int>::MemLocation::UNIFIED_MEM> um_neighbors;
#endif
#endif
#if defined(HAVE_GPU)
#if DATAMEMBER_ARRAY_ALLOCATION == OPTION_4b
      int um_neighbors_max_elements;
#endif
#endif
#if defined(HAVE_GPU)
#if DATAMEMBER_ARRAY_ALLOCATION == OPTION_4b
      ShallowArray_Flat<< int* , Array_Flat<int>::MemLocation::UNIFIED_MEM> um_neighbors_num_elements;
#endif
#endif
      static CG_LifeNodeSharedMembers* CG_sharedMembers;
   protected:
#if defined(HAVE_MPI)
      std::map <int, CCDemarshaller*> _demarshallerMap;
#endif
#if defined(HAVE_MPI)
      std::map <int, CCDemarshaller*>::iterator _demarshallerMapIter;
#endif
#if defined(HAVE_MPI)
      std::map <int, ShallowArray<CG_LifeNode*> > _sendMap;
#endif
#if defined(HAVE_MPI)
      std::map <int, ShallowArray<CG_LifeNode*> >::iterator _sendMapIter;
#endif
#if ! defined(HAVE_GPU)
      ShallowArray<LifeNode, 1000, 4> _nodes;
#endif
#if defined(HAVE_GPU)
      ShallowArray_Flat<LifeNode, Array_Flat<int>::MemLocation::CPU, 1000> _nodes;
#endif
      ConnectionIncrement _computeCost;
   private:
#if defined(HAVE_MPI)
      std::map<std::string, CG_T_SendFunctionPtr> CG_sendTemplates;
#endif
#if defined(HAVE_MPI)
      std::map<std::string, CG_T_GetSendTypeFunctionPtr> CG_getSendTypeTemplates;
#endif
};

#endif
