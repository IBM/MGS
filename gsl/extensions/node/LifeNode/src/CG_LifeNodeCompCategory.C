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

#include "Lens.h"
#include "CG_LifeNodeCompCategory.h"
#include "LifeNodeCompCategory.h"
#include "CG_LifeNode.h"
#include "NDPairList.h"
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
#include <numeric>
#include <cmath>

#ifdef TEST_USING_GPU_COMPUTING
#include "LifeNodeCompCategory.cu"
#endif
#ifdef TEST_USING_GPU_COMPUTING
#include "LifeNodeCompCategory.incl"
#endif


CG_LifeNodeCompCategory::CG_LifeNodeCompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList) 
   : NodeCompCategoryBase(sim, modelName)
{
   if (CG_sharedMembers==0) {
      CG_sharedMembers = new CG_LifeNodeSharedMembers();
      CG_sharedMembers->setUp(ndpList);
   }
   initializePhase("initialize", "Init", false);
   initializePhase("update", "Runtime", false);
   initializePhase("copy", "Runtime", true);
}

void CG_LifeNodeCompCategory::getInitializationParameterSet(std::unique_ptr<ParameterSet>& initPSet) 
{
   initPSet.reset(new CG_LifeNodePSet());
}

void CG_LifeNodeCompCategory::getInAttrParameterSet(std::unique_ptr<ParameterSet>& CG_castedPSet) 
{
   CG_castedPSet.reset(new CG_LifeNodeInAttrPSet());
}

void CG_LifeNodeCompCategory::getOutAttrParameterSet(std::unique_ptr<ParameterSet>& CG_castedPSet) 
{
   CG_castedPSet.reset(new CG_LifeNodeOutAttrPSet());
}

void CG_LifeNodeCompCategory::CG_InstancePhase_initialize(NodePartitionItem* arg, CG_LifeNodeWorkUnitInstance* wu) 
{
#if defined(HAVE_GPU) 
   ShallowArray_Flat<LifeNode>::iterator it = _nodes.begin();
   ShallowArray_Flat<LifeNode>::iterator end = _nodes.begin();
#else
   ShallowArray<LifeNode>::iterator it = _nodes.begin();
   ShallowArray<LifeNode>::iterator end = _nodes.begin();
#endif
   it += arg->startIndex;
   end += arg->endIndex;
   for (; it <= end; ++it) {
     (*it).initialize(wu->getRNG());
   }
}

void CG_LifeNodeCompCategory::CG_InstancePhase_update(NodePartitionItem* arg, CG_LifeNodeWorkUnitInstance* wu) 
{
#if defined(HAVE_GPU) 
   ShallowArray_Flat<LifeNode>::iterator it = _nodes.begin();
   ShallowArray_Flat<LifeNode>::iterator end = _nodes.begin();
#else
   ShallowArray<LifeNode>::iterator it = _nodes.begin();
   ShallowArray<LifeNode>::iterator end = _nodes.begin();
#endif
   it += arg->startIndex;
   end += arg->endIndex;
   for (; it <= end; ++it) {
     (*it).update(wu->getRNG());
   }
}

void CG_LifeNodeCompCategory::CG_InstancePhase_copy(NodePartitionItem* arg, CG_LifeNodeWorkUnitInstance* wu) 
{
#if defined(HAVE_GPU) 
   ShallowArray_Flat<LifeNode>::iterator it = _nodes.begin();
   ShallowArray_Flat<LifeNode>::iterator end = _nodes.begin();
#else
   ShallowArray<LifeNode>::iterator it = _nodes.begin();
   ShallowArray<LifeNode>::iterator end = _nodes.begin();
#endif
   it += arg->startIndex;
   end += arg->endIndex;
   for (; it <= end; ++it) {
      (*it).copy(wu->getRNG());
   }
}

void CG_LifeNodeCompCategory::getWorkUnits() 
{
  
   switch(_sim.getPhaseMachineType("initialize"))
   {
     case machineType::GPU :
     {
      NodePartitionItem* it = _GPUpartitions;
      NodePartitionItem* end = it + _nbrGPUpartitions;
      for (; it < end; ++it) {
         WorkUnit* workUnit = new CG_LifeNodeWorkUnitInstance(it, &CG_LifeNodeCompCategory::CG_host_initialize, this);
         _workUnits["initialize"].push_back(workUnit);
      }
      _sim.addWorkUnits(getSimulationPhaseName("initialize"), _workUnits["initialize"] );
     }
     break;

     case machineType::CPU :
     {
      NodePartitionItem* it = _CPUpartitions;
      NodePartitionItem* end = it + _nbrCPUpartitions;
      for (; it < end; ++it) {
         WorkUnit* workUnit = new CG_LifeNodeWorkUnitInstance(it, &CG_LifeNodeCompCategory::CG_InstancePhase_initialize, this);
         _workUnits["initialize"].push_back(workUnit);
      }
      _sim.addWorkUnits(getSimulationPhaseName("initialize"), _workUnits["initialize"] );
     }
     break;

     default :
      assert(0);
      break;
   }
   
   switch(_sim.getPhaseMachineType("update"))
   {
     case machineType::GPU :
     {
      NodePartitionItem* it = _GPUpartitions;
      NodePartitionItem* end = it + _nbrGPUpartitions;
      for (; it < end; ++it) {
         WorkUnit* workUnit = new CG_LifeNodeWorkUnitInstance(it, &CG_LifeNodeCompCategory::CG_host_update, this);
         _workUnits["update"].push_back(workUnit);
      }
      _sim.addWorkUnits(getSimulationPhaseName("update"), _workUnits["update"] );
     }
     break;

     case machineType::CPU :
     {
      NodePartitionItem* it = _CPUpartitions;
      NodePartitionItem* end = it + _nbrCPUpartitions;
      for (; it < end; ++it) {
         WorkUnit* workUnit = new CG_LifeNodeWorkUnitInstance(it, &CG_LifeNodeCompCategory::CG_InstancePhase_update, this);
         _workUnits["update"].push_back(workUnit);
      }
      _sim.addWorkUnits(getSimulationPhaseName("update"), _workUnits["update"] );
     }
     break;
      
     default :
      assert(0);
      break;
   }

   
   switch(_sim.getPhaseMachineType("copy"))
   {
     case machineType::GPU :
     {
      NodePartitionItem* it = _GPUpartitions;
      NodePartitionItem* end = it + _nbrGPUpartitions;
      for (; it < end; ++it) {
         WorkUnit* workUnit = new CG_LifeNodeWorkUnitInstance(it, &CG_LifeNodeCompCategory::CG_host_copy, this);
         _workUnits["copy"].push_back(workUnit);
      }
      _sim.addWorkUnits(getSimulationPhaseName("copy"), _workUnits["copy"] );
     }
     break;

     case machineType::CPU :
     {
      NodePartitionItem* it = _CPUpartitions;
      NodePartitionItem* end = it + _nbrCPUpartitions;
      for (; it < end; ++it) {
	WorkUnit* workUnit = new CG_LifeNodeWorkUnitInstance(it, &CG_LifeNodeCompCategory::CG_InstancePhase_copy, this);
	_workUnits["copy"].push_back(workUnit);
      }
      _sim.addWorkUnits(getSimulationPhaseName("copy"), _workUnits["copy"] );
     }
     break;

     default :
      assert(0);
      break;
   }
}

#ifdef HAVE_MPI
void CG_LifeNodeCompCategory::addToSendMap(int toPartitionId, Node* node) 
{
   CG_LifeNode* localNode = dynamic_cast<CG_LifeNode*>(node);
   assert(localNode);
   ShallowArray<CG_LifeNode*>::iterator it = _sendMap[toPartitionId].begin();
   ShallowArray<CG_LifeNode*>::iterator end = _sendMap[toPartitionId].end();
   bool found = false;
   for (; it != end; ++it) {
      if ((*it) == localNode) {
         found = true;
         break;
      }
   }
   if (found == false)
      _sendMap[toPartitionId].push_back(localNode);
}
#endif

#ifdef HAVE_MPI
void CG_LifeNodeCompCategory::allocateProxy(int fromPartitionId, NodeDescriptor* nd) 
{
#if defined(HAVE_GPU) 
   #if PROXY_ALLOCATION == OPTION_3
      /* local proxy data + local _receiveList */
      CCDemarshaller* ccd = findDemarshaller(fromPartitionId);
      NodeProxyBase* proxy = ccd->addDestination();
      proxy->setNodeDescriptor(nd);
      ((CG_LifeNodeProxy*)proxy)->setCompCategory(ccd->_receiveList.size()-1, this, fromPartitionId);
      nd->setNode(proxy);
   #elif PROXY_ALLOCATION == OPTION_4
      CCDemarshaller* ccd = findDemarshaller(fromPartitionId);
      /* global proxy data */
      int sz = proxy_um_value.size()+1;
      public_um_value.increaseSizeTo(sz);
      public_um_publicValue.increaseSizeTo(sz);
      public_um_neighbors.increaseSizeTo(sz);
      int MAX_SUBARRAY_SIZE = 20;
      //NOTE: um_neighbors is an array of array
      public_um_neighbors[sz-1].resize_allocated_subarray(MAX_SUBARRAY_SIZE, Array_Flat<int>::MemLocation::UNIFIED_MEM);

      /* local _receiveList */
      NodeProxyBase* proxy = ccd->addDestination();
      proxy->setNodeDescriptor(nd);
      ((CG_LifeNodeProxy*)proxy)->setCompCategory(sz-1, this);
      nd->setNode(proxy);
   #endif
   
#else
   CCDemarshaller* ccd = findDemarshaller(fromPartitionId);
   NodeProxyBase* proxy = ccd->addDestination();
   proxy->setNodeDescriptor(nd);
   nd->setNode(proxy);
#endif
}
#endif

#ifdef HAVE_MPI
void CG_LifeNodeCompCategory::addVariableNamesForPhase(std::set<std::string>& namesSet, const std::string& phase) 
{
   if (phase == "copy") {
      namesSet.insert("publicValue");
   }
}
#endif

#ifdef HAVE_MPI
void CG_LifeNodeCompCategory::setDistributionTemplates() 
{
   CG_sendTemplates["FLUSH_LENS"] = &CG_LifeNode::CG_send_FLUSH_LENS;
   CG_getSendTypeTemplates["FLUSH_LENS"] = &CG_LifeNode::CG_getSendType_FLUSH_LENS;
   std::map<std::string, Phase*>::iterator it, end = _phaseMappings.end();
   for (it = _phaseMappings.begin(); it != end; ++it) {
      if (it->second->getName() == getSimulationPhaseName("copy")){
         CG_sendTemplates[it->second->getName()] = &CG_LifeNode::CG_send_copy;
         CG_getSendTypeTemplates[it->second->getName()] = &CG_LifeNode::CG_getSendType_copy;
      }
   }
}
#endif

#ifdef HAVE_MPI
void CG_LifeNodeCompCategory::resetSendProcessIdIterators() 
{
   _sendMapIter=_sendMap.begin();
}
#endif

#ifdef HAVE_MPI
int CG_LifeNodeCompCategory::getSendNextProcessId() 
{
   int rval=-1;
   if (_sendMapIter!=_sendMap.end()) rval=(*_sendMapIter).first;
   ++_sendMapIter;
   return rval;
}
#endif

#ifdef HAVE_MPI
bool CG_LifeNodeCompCategory::atSendProcessIdEnd() 
{
   return (_sendMapIter==_sendMap.end());
}
#endif

#ifdef HAVE_MPI
void CG_LifeNodeCompCategory::resetReceiveProcessIdIterators() 
{
   _demarshallerMapIter=_demarshallerMap.begin();
}
#endif

#ifdef HAVE_MPI
int CG_LifeNodeCompCategory::getReceiveNextProcessId() 
{
   int rval=-1;
   if (_demarshallerMapIter!=_demarshallerMap.end()) rval=(*_demarshallerMapIter).first;
   ++_demarshallerMapIter;
   return rval;
}
#endif

#ifdef HAVE_MPI
bool CG_LifeNodeCompCategory::atReceiveProcessIdEnd() 
{
   return (_demarshallerMapIter==_demarshallerMap.end());
}
#endif

#ifdef HAVE_MPI
int CG_LifeNodeCompCategory::setMemPattern(std::string phaseName, int dest, MemPattern* mpptr) 
{
   std::map<std::string, CG_T_GetSendTypeFunctionPtr>::iterator fiter = CG_getSendTypeTemplates.find(phaseName);
   int nBytes=0;
   bool inList=(fiter != CG_getSendTypeTemplates.end());
   if (inList) {
      ShallowArray<CG_LifeNode*> &nodes = _sendMap[dest];
      inList = inList && (nodes.size()!=0);
      if (inList) {
         ShallowArray<CG_LifeNode*>::iterator niter = nodes.begin();
         std::vector<int> npls;
         std::vector<MPI_Aint> blocs;
         CG_T_GetSendTypeFunctionPtr & function = (fiter->second);
         ((*niter)->*(function))(npls, blocs);
         int npblocks=npls.size();
         assert(npblocks==blocs.size());
         inList = inList && (npblocks!=0);
         if (inList) {
            std::vector<int> nplengths;
            std::vector<int> npdispls;
            MPI_Aint nodeAddress;
            MPI_Get_address(*niter, &nodeAddress);
            mpptr->orig = reinterpret_cast<char*>(*niter);
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
            int nblocks=nodes.size();
            nBytes *= nblocks;
            MPI_Aint naddr, prevnaddr;
            MPI_Get_address(*niter, &prevnaddr);
            int* bdispls = mpptr->allocateOrigDispls(nblocks);
            bdispls[0]=0;
            ++niter;
            for (int i=1; i<nblocks; ++i, ++niter) {
               MPI_Get_address(*niter, &naddr);
               bdispls[i]=naddr-prevnaddr;
               prevnaddr=naddr;
            }
         }
      }
   }
   return nBytes;
}
#endif

#ifdef HAVE_MPI
int CG_LifeNodeCompCategory::getIndexedBlock(std::string phaseName, int dest, MPI_Datatype* blockType, MPI_Aint& blockLocation) 
{
   std::map<std::string, CG_T_GetSendTypeFunctionPtr>::iterator fiter = CG_getSendTypeTemplates.find(phaseName);
   int nBytes=0;
   bool inList=(fiter != CG_getSendTypeTemplates.end());
   if (inList) {
      ShallowArray<CG_LifeNode*> &nodes = _sendMap[dest];
      inList = inList && (nodes.size()!=0);
      if (inList) {
         ShallowArray<CG_LifeNode*>::iterator niter = nodes.begin();
         std::vector<int> npls;
         std::vector<MPI_Aint> blocs;
         CG_T_GetSendTypeFunctionPtr & function = (fiter->second);
         ((*niter)->*(function))(npls, blocs);
         int npblocks=npls.size();
         assert(npblocks==blocs.size());
         inList = inList && (npblocks!=0);
         if (inList) {
            int* nplengths = new int[npblocks];
            int* npdispls = new int[npblocks];
            MPI_Aint nodeAddress;
            MPI_Get_address(*niter, &nodeAddress);
            for (int i=0; i<npblocks; ++i) {
               nplengths[i]=npls[i];
               npdispls[i]=blocs[i]-nodeAddress;
            }
            MPI_Datatype nodeTypeBasic, nodeType;
            MPI_Type_indexed(npblocks, nplengths, npdispls, MPI_CHAR, &nodeTypeBasic);
            MPI_Type_create_resized(nodeTypeBasic, 0, sizeof(CG_LifeNode), &nodeType);
            delete [] nplengths;
            delete [] npdispls;

            int nblocks=nodes.size();
            int* blengths = new int[nblocks];
            MPI_Aint* bdispls = new MPI_Aint[nblocks];
            blockLocation=nodeAddress;
            for (int i=0; i<nblocks; ++i, ++niter) {
               npls.clear();
               blocs.clear();
               ((*niter)->*(function))(npls, blocs);
               npblocks=npls.size();
               for (int j=0; j<npblocks; ++j) nBytes += npls[j];
               blengths[i]=1;
               MPI_Get_address(*niter, &bdispls[i]);
               bdispls[i]-=blockLocation;
            }
            MPI_Type_create_hindexed(nblocks, blengths, bdispls, nodeType, blockType);
            MPI_Type_free(&nodeType);
            delete [] blengths;
            delete [] bdispls;
         }
      }
   }
   return nBytes;
}
#endif

#ifdef HAVE_MPI
IndexedBlockCreator* CG_LifeNodeCompCategory::getReceiveBlockCreator(int fromPartitionId) 
{
   return getDemarshaller(fromPartitionId);
}
#endif

#ifdef HAVE_MPI
CG_LifeNodeCompCategory::CCDemarshaller* CG_LifeNodeCompCategory::getDemarshaller(int fromPartitionId) 
{
   CCDemarshaller* ccd=0;
   std::map <int, CCDemarshaller*>::iterator iter;
   iter = _demarshallerMap.find(fromPartitionId);
   if (iter != _demarshallerMap.end()) {
      ccd = (*iter).second;
   }
   return ccd;
}
#endif

#ifdef HAVE_MPI
CG_LifeNodeCompCategory::CCDemarshaller* CG_LifeNodeCompCategory::findDemarshaller(int fromPartitionId) 
{
   CCDemarshaller* ccd;
   std::map <int, CCDemarshaller*>::iterator iter;
   iter = _demarshallerMap.find(fromPartitionId);
   if (iter == _demarshallerMap.end()) {
      ccd = new CCDemarshaller(&getSimulation());
      _demarshallerMap[fromPartitionId] = ccd;

      std::unique_ptr<CG_LifeNodeProxyDemarshaller> ap;
      CG_LifeNodeProxy::CG_recv_FLUSH_LENS_demarshaller(ap);
      ccd->CG_recvTemplates["FLUSH_LENS"] = ap.release();

      std::map<std::string, Phase*>::iterator it, end = _phaseMappings.end();
      for (it = _phaseMappings.begin(); it != end; ++it) {
         if (it->second->getName() == getSimulationPhaseName("copy")){
            std::unique_ptr<CG_LifeNodeProxyDemarshaller> ap;
            CG_LifeNodeProxy::CG_recv_copy_demarshaller(ap);
            ccd->CG_recvTemplates[(it->second->getName())] = ap.release();
         }
      }
   } else {
      ccd = (*iter).second;
   }
   return ccd;
}
#endif

CG_LifeNodeSharedMembers& CG_LifeNodeCompCategory::getSharedMembers() 
{
   return *CG_sharedMembers;
}

void CG_LifeNodeCompCategory::getNodeAccessor(std::unique_ptr<NodeAccessor>& nodeAccessor, GridLayerDescriptor* gridLayerDescriptor) 
{
   CG_LifeNodeGridLayerData* currentGridLayerData = new CG_LifeNodeGridLayerData(this, gridLayerDescriptor, _gridLayerDataArraySize);
   _gridLayerDataList.push_back(currentGridLayerData);
   _gridLayerDataArraySize++;
   _gridLayerDataOffsets.push_back(_gridLayerDataOffsets.back()+currentGridLayerData->getNbrUnits());
   nodeAccessor.reset(new CG_LifeNodeNodeAccessor(getSimulation(), gridLayerDescriptor, currentGridLayerData));
}

void CG_LifeNodeCompCategory::allocateNode(NodeDescriptor* nd) 
{
   _nodes.increaseSizeTo(_nodes.size()+1);
#if defined(HAVE_GPU) 
   int sz = _nodes.size();
   //TUAN TODO
   //// maybe we want CG_LifeNode to have a static (i.e. single copy)
   // data member _container, as there is actually only 1 copy of LifeNodeCompCategory
   _nodes[_nodes.size()-1].setCompCategory(_nodes.size()-1, this);
   um_value.increaseSizeTo(sz);
   um_publicValue.increaseSizeTo(sz);

#if DATAMEMBER_ARRAY_ALLOCATION == OPTION_3
   //NODE for array, we may want to consider allocate the internal array 
   //// with large enough size
   //by checking of type is a pointer
   um_neighbors.increaseSizeTo(sz);
   int MAX_SUBARRAY_SIZE = 20;
   //NOTE: um_neighbors is an array of array
   //TUAN
   ////This is the one that hurt the performance of initialization
   um_neighbors[sz-1].resize_allocated_subarray(MAX_SUBARRAY_SIZE, Array_Flat<int>::MemLocation::UNIFIED_MEM);
#elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_4
   int MAX_SUBARRAY_SIZE = 20;
   um_neighbors.increaseSizeTo(sz*MAX_SUBARRAY_SIZE);
   um_neighbors_start_offset.increaseSizeTo(sz);
   um_neighbors_num_elements.increaseSizeTo(sz);
   um_neighbors_start_offset[sz-1] = (sz-1) * MAX_SUBARRAY_SIZE;
   um_neighbors_num_elements[sz-1] = 0; //TUAN TODO consider having an API to ShallowArray to set all values
   // which internally calls cudaMemSet or an equivalent functions
#elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_4b
   um_neighbors.increaseSizeTo(sz*um_neighbors_max_elements);
   um_neighbors_num_elements.increaseSizeTo(sz);
   um_neighbors_num_elements[sz-1] = 0; //TUAN TODO consider having an API to ShallowArray to set all values
   // which internally calls cudaMemSet or an equivalent functions
#elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_5
   //do something here
   //int MAX_SUBARRAY_SIZE = 20;
   um_neighbors.increaseSizeTo(sz*MAX_SUBARRAY_SIZE);
   um_neighbors_start_offset.increaseSizeTo(sz);
   // NOTE: We need an internal data structure to keep tracks how many incoming connections to a given 'reference' data array 
   // TUAN TODO fix this equation
   um_neighbors_start_offset[sz-1] = (sz-1) * MAX_SUBARRAY_SIZE;
#endif
#endif
   _nodes[_nodes.size()-1].setNodeDescriptor(nd);
   nd->setNode(&_nodes[_nodes.size()-1]);
   nd->getGridLayerData()->incrementNbrNodesAllocated();
}

void CG_LifeNodeCompCategory::allocateNodes(size_t size) 
{
#if defined(HAVE_GPU) 
   bool force_resize = true;
   _nodes.resize_allocated(size, force_resize);
   um_value.resize_allocated(size, force_resize);
   um_publicValue.resize_allocated(size, force_resize);
   /*
    * MDL:   int*[] neighbors;
    * C++:
    *     ShallowArray<  ShallowArray<int*> > um_neighbors;l
    */
   //NODE for array, we may want to consider allocate the internal array 
   //// with large enough size
   //by checking of type is a pointer
   //NOTE: We need to make a choice
   // 1. orgnanize as single-large 
   // *     ShallowArray<int*> um_neighbors;
   // with elements from 
   //  um_neighbors[(i-1)* MAX_SUBARRAY_SIZE : i * MAX_SUBARRAY_SIZE-1]
   //     are where elements from node[i]'s um_neighbors
   // new_size = size * MAX_SUBARRAY_SIZE
   //um_neighbors.resize_allocated(size, MAX_SUBARRAY_SIZE, force_resize);
   // 2. orgnanize as array of array
   // * C++:
   // *     ShallowArray<  ShallowArray<int*> > um_neighbors;l
   // with elements from 
   //  [(i-1)* MAX_SUBARRAY_SIZE : i * MAX_SUBARRAY_SIZE-1]
   //     are where elements from um_neighbors[i] reside 
   //   This requires modify the allocator, so that it first allocated large enough memory
   
   //    and have a check inside ::resize_allocated
   //    of non-zero _num
   //um_neighbors.set_preallocated_size(MAX_SUBARRAY_SIZE);
   //um_neighbors.resize_allocated_array(size, force_resize, MAX_SUBARRAY_SIZE);
   //um_neighbors.resize_allocated_array(size, force_resize, MAX_SUBARRAY_SIZE);
   // 3. orgnanize as array of array
   //     ShallowArray<ShallowArray<T>>
#if DATAMEMBER_ARRAY_ALLOCATION == OPTION_3
   //do nothing
   um_neighbors.resize_allocated(size, force_resize);
   //NOTE: we put the code inside :allocateNode
   //  and potentially ::allocateProxy
   //int MAX_SUBARRAY_SIZE = 20;
   ////if (isDerivedFrom<Array_Flat, decltype(um_neighbors[0])>::value)
   //{
   //   for (int i = 0; i < size; i++)
   //      um_neighbors[i].resize_allocated_subarray(MAX_SUBARRAY_SIZE);
   //}
#elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_4
   //do something here
   um_neighbors.resize_allocated(size*MAX_SUBARRAY_SIZE, force_resize);
   um_neighbors_start_offset.resize_allocated(size, force_resize);
   um_neighbors_num_elements.resize_allocated(size, force_resize);
#elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_4b
   //do something here
   int MAX_SUBARRAY_SIZE = 20;
   um_neighbors_max_elements = MAX_SUBARRAY_SIZE;
   um_neighbors.resize_allocated(size*um_neighbors_max_elements, force_resize);
   um_neighbors_num_elements.resize_allocated(size, force_resize);
#elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_5
   //do something here
   int um_neighbors_total_connections=0;
   // NOTE: We need an internal data structure to keep tracks how many incoming connections to a given 'reference' data array 
   /*
    for each node
    {
           um_neighbors_total_connection += node[i].num_input_connections_for(um_neighbors);
    }
    */
   um_neighbors.resize_allocated(um_neighbors_total_connection, force_resize);
#endif
#endif
}

void CG_LifeNodeCompCategory::allocateProxies(const std::vector<int>& sizes)
{
#if defined(HAVE_GPU) 
   unsigned my_rank = _sim.getRank();
   bool force_resize = true;
#if PROXY_ALLOCATION == OPTION_3
   for (int i = 0; i < _sim.getNumProcesses(); i++)
   {
      if (i != my_rank)
      {
         CCDemarshaller* ccd = findDemarshaller(i);
         int size = sizes[i];
         if (size > 0)
         {
            //kv.second->allocateProxies(kv.first, sizes[kv.first])
            ccd->_receiveList.resize_allocated(size, force_resize);
            ccd->um_value.resize_allocated(size, force_resize);
            ccd->um_publicValue.resize_allocated(size, force_resize);
            ccd->um_neighbors.resize_allocated(size, force_resize);
         }
      }
   }
   //for (auto& kv : _demarshallerMap)
   //{
   //   if (kv.first != my_rank)
   //   {
   //      CCDemarshaller* ccd = findDemarshaller(kv.first);
   //      int size = sizes[kv.first];
   //      //kv.second->allocateProxies(kv.first, sizes[kv.first])
   //      ccd->_receiveList.resize_allocated(size, force_resize);
   //      ccd->um_value.resize_allocated(size, force_resize);
   //      ccd->um_publicValue.resize_allocated(size, force_resize);
   //      ccd->um_neighbors.resize_allocated(size, force_resize);
   //   }
   //}
#elif PROXY_ALLOCATION == OPTION_4
   int total = std::accumulate(sizes.begin(), sizes.end(), 0); 
   assert(0);
   int offset = 0;
   for (int i = 0; i < _sim.getNumProcesses(); i++)
   {
      offset += sizes[i];
      if (i != my_rank)
      {
         CCDemarshaller* ccd = findDemarshaller(i);
         int size = sizes[i];
         if (size > 0)
         {
            ccd->_receiveList.resize_allocated(size, force_resize);
            ccd->offset = offset;
         }
      }
   }
   this->proxy_um_value.resize_allocated(total, force_resize);
   this->proxy_um_publicValue.resize_allocated(total, force_resize);
   this->proxy_um_neighbors.resize_allocated(total, force_resize);
#endif

#endif
}

int CG_LifeNodeCompCategory::getNbrComputationalUnits() 
{
   return _nodes.size();
}

ConnectionIncrement* CG_LifeNodeCompCategory::getComputeCost() 
{
   return &_computeCost;
}

CG_LifeNodeCompCategory::~CG_LifeNodeCompCategory() 
{
#ifdef HAVE_MPI
   std::map<int, CCDemarshaller*>::iterator end2 = _demarshallerMap.end();
   for (std::map<int, CCDemarshaller*>::iterator iter2=_demarshallerMap.begin(); iter2!=end2; ++iter2) {
      delete (*iter2).second;
   }
#endif
   if (CG_sharedMembers) {
      delete CG_sharedMembers;
      CG_sharedMembers=0;
   }
}

CG_LifeNodeSharedMembers* CG_LifeNodeCompCategory::CG_sharedMembers = 0;
