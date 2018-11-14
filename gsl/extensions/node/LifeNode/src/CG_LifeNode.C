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
#include "CG_LifeNode.h"
#include "CG_LifeNodeTriggerableCaller.h"
#include "CG_LifeNodeCompCategory.h"
#include "ConnectionIncrement.h"
#include "Constant.h"
#include "Edge.h"
#include "Node.h"
#include "Variable.h"
#include "VariableDescriptor.h"
#include "CG_LifeNodeInAttrPSet.h"
#include "CG_LifeNodeOutAttrPSet.h"
#include "CG_LifeNodePSet.h"
#ifdef HAVE_MPI
#include "CG_LifeNodeProxyDemarshaller.h"
#endif
#include "CG_LifeNodePublisher.h"
#include "CG_LifeNodeSharedMembers.h"
#include "DataItem.h"
#include "DataItemArrayDataItem.h"
#include "IntArrayDataItem.h"
#include "IntDataItem.h"
#ifdef HAVE_MPI
#include "Marshall.h"
#endif
#include "NodeBase.h"
#ifdef HAVE_MPI
#include "OutputStream.h"
#endif
#include "Service.h"
#include "ShallowArray.h"
#include "String.h"
#include "SyntaxErrorException.h"
#include "ValueProducer.h"
#include "VariableDescriptor.h"
#include <iostream>
#include <memory>
#include <algorithm>



int* CG_LifeNode::CG_get_ValueProducer_value() 
{
#if defined(HAVE_GPU) 
   return &(_container->um_publicValue[index]);
#else
   return &publicValue;
#endif
}

const char* CG_LifeNode::getServiceName(void* data) const
{
#if defined(HAVE_GPU) 
   if (data == &(_container->um_value[index])) {
      return "value";
   }
   if (data == &(_container->um_publicValue[index])) {
      return "publicValue";
   }
 #if DATAMEMBER_ARRAY_ALLOCATION == OPTION_3
   if (data == &(_container->um_neighbors[index])) {
      return "neighbors";
   }
 #elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_4
   if (data == &(_container->um_neighbors[_container->um_neighbors_start_offset[index]])) {
      return "neighbors";
   }
 #elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_4b
   if (data == &(_container->um_neighbors[index*_container->um_neighbors_max_elements])) {
      return "neighbors";
   }
 #elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_5
   assert(0);
 #endif
   if (data == &(getSharedMembers().tooCrowded)) {
      return "tooCrowded";
   }
   if (data == &(getSharedMembers().tooSparse)) {
      return "tooSparse";
   }
#else
   if (data == &(value)) {
      return "value";
   }
   if (data == &(publicValue)) {
      return "publicValue";
   }
   if (data == &(neighbors)) {
      return "neighbors";
   }
   if (data == &(getSharedMembers().tooCrowded)) {
      return "tooCrowded";
   }
   if (data == &(getSharedMembers().tooSparse)) {
      return "tooSparse";
   }
#endif
   return "Error in Service Name!";
}

const char* CG_LifeNode::getServiceDescription(void* data) const
{
   //IMPORTANT: By default, non of the data member is a service
   // i.e. we cannot use via GSL
#if defined(HAVE_GPU) 
   if (data == &(_container->um_value[index])) {
      return "";
   }
   if (data == &(_container->um_publicValue[index])) {
      return "";
   }
 #if DATAMEMBER_ARRAY_ALLOCATION == OPTION_3
   if (data == &(_container->um_neighbors[index])) {
      return "";
   }
 #elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_4
   if (data == &(_container->um_neighbors[_container->um_neighbors_start_offset[index]])) {
      return "";
   }
 #elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_4
   if (data == &(_container->um_neighbors[index*_container->um_neighbors_max_elements])) {
      return "";
   }
 #elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_5
   assert(0);
 #endif
   if (data == &(getSharedMembers().tooCrowded)) {
      return "";
   }
   if (data == &(getSharedMembers().tooSparse)) {
      return "";
   }
#else
   if (data == &(value)) {
      return "";
   }
   if (data == &(publicValue)) {
      return "";
   }
   if (data == &(neighbors)) {
      return "";
   }
   if (data == &(getSharedMembers().tooCrowded)) {
      return "";
   }
   if (data == &(getSharedMembers().tooSparse)) {
      return "";
   }
#endif
   return "Error in Service Description!";
}

Publisher* CG_LifeNode::getPublisher() 
{
   if (_publisher == 0) {
      _publisher = new CG_LifeNodePublisher(getSimulation(), this);
   }
   return _publisher;
}

#ifdef HAVE_MPI
void CG_LifeNode::CG_getSendType_copy(std::vector<int>& blengths, std::vector<MPI_Aint>& blocs) const
{
   MarshallerInstance<int > mi0;
#if defined(HAVE_GPU) 
   //TUAN TODO - consider to revise this
   //  as we no longer need block-length, and block-index
   //  in a flat array
   mi0.getBlocks(blengths, blocs, _container->um_publicValue[index]);
#else
   mi0.getBlocks(blengths, blocs, publicValue);
#endif
}
#endif

#ifdef HAVE_MPI
void CG_LifeNode::CG_send_FLUSH_LENS(OutputStream* stream) const
{
   MarshallerInstance<int > mi0;
#if defined(HAVE_GPU) 
   mi0.marshall(stream, _container->um_publicValue[index]);
#else
   mi0.marshall(stream, publicValue);
#endif
}
#endif

#ifdef HAVE_MPI
void CG_LifeNode::CG_getSendType_FLUSH_LENS(std::vector<int>& blengths, std::vector<MPI_Aint>& blocs) const
{
   MarshallerInstance<int > mi0;
#if defined(HAVE_GPU) 
   //TUAN TODO - consider to revise this
   //  as we no longer need block-length, and block-index
   //  in a flat array
   mi0.getBlocks(blengths, blocs, _container->um_publicValue[index]);
#else
   mi0.getBlocks(blengths, blocs, publicValue);
#endif
}
#endif

void CG_LifeNode::initialize(ParameterSet* CG_initPSet) 
{
   //TUAN NOTE: 
   //This method is internally evoked by 'InitNode' statement in GSL
   // Using 'InitNode' is not that efficient to initialize data
   // Please consider using a InitPhase's CG_host_initialize() and evoke the kernel 
   CG_LifeNodePSet* CG_pset = dynamic_cast<CG_LifeNodePSet*>(CG_initPSet);
#if defined(HAVE_GPU) 
   //value() = CG_pset->value;
   //publicValue() = CG_pset->publicValue;
   //neighbors() = CG_pset->neighbors;
   _container->um_value[index] = CG_pset->value;
   _container->um_publicValue[index] = CG_pset->publicValue;
 #if DATAMEMBER_ARRAY_ALLOCATION == OPTION_3
   _container->um_neighbors[index] = CG_pset->neighbors;
 #elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_4
   auto um_neighbors_from = _container->um_neighbors_start_offset[index];
   auto um_neighbors_to = _container->um_neighbors_start_offset[index+1]-1;
   //TUAN TODO : implement SliceArray 
   //_container->um_neighbors[um_neighbors_from : um_neighbors_to] = CG_pset->neighbors;
   // ... or for now
   for (auto i = 0; i <  std::min(um_neighbors_to - um_neighbors_from+1, (int)CG_pset->neighbors.size()); ++i)
   {
      _container->um_neighbors[i+um_neighbors_from] = CG_pset->neighbors[i];
   }
 #elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_4b
   auto um_neighbors_from =  index * _container->um_neighbors_max_elements;
   auto um_neighbors_to =  (index+1) * _container->um_neighbors_max_elements - 1;
   //TUAN TODO : implement SliceArray 
   //_container->um_neighbors[um_neighbors_from : um_neighbors_to] = CG_pset->neighbors;
   // ... or for now
   for (auto i = 0; i <  std::min(um_neighbors_to - um_neighbors_from+1, (int)CG_pset->neighbors.size()); ++i)
   {
      _container->um_neighbors[i+um_neighbors_from] = CG_pset->neighbors[i];
   }
 #elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_5
   //_container->um_neighbors[index] = CG_pset->neighbors;
   assert(0);
 #endif

#else
   value = CG_pset->value;
   publicValue = CG_pset->publicValue;
   neighbors = CG_pset->neighbors;
#endif
}

TriggerableBase::EventType CG_LifeNode::createTriggerableCaller(const std::string& CG_triggerableFunctionName, NDPairList* CG_triggerableNdpList, std::unique_ptr<TriggerableCaller>& CG_triggerableCaller) 
{
   throw SyntaxErrorException(CG_triggerableFunctionName + " is not defined in LifeNode as a Triggerable function.");
   return TriggerableBase::_UNALTERED;
}

void CG_LifeNode::getInitializationParameterSet(std::unique_ptr<ParameterSet>& initPSet) const
{
   initPSet.reset(new CG_LifeNodePSet());
}

void CG_LifeNode::getInAttrParameterSet(std::unique_ptr<ParameterSet>& CG_castedPSet) const
{
   CG_castedPSet.reset(new CG_LifeNodeInAttrPSet());
}

void CG_LifeNode::getOutAttrParameterSet(std::unique_ptr<ParameterSet>& CG_castedPSet) const
{
   CG_castedPSet.reset(new CG_LifeNodeOutAttrPSet());
}

void CG_LifeNode::acceptService(Service* service, const std::string& name) 
{
   if (name == "value") {
      GenericService< int >* CG_local = dynamic_cast<GenericService< int >*>(service);
      if (CG_local == 0) {
         throw SyntaxErrorException("Expected a int service for value");
      }
#if defined(HAVE_GPU) 
      _container->um_value[index] = *CG_local->getData();
#else
      value = *CG_local->getData();
#endif
      return;
   }
   if (name == "publicValue") {
      GenericService< int >* CG_local = dynamic_cast<GenericService< int >*>(service);
      if (CG_local == 0) {
         throw SyntaxErrorException("Expected a int service for publicValue");
      }
#if defined(HAVE_GPU) 
      _container->um_publicValue[index] = *CG_local->getData();
#else
      publicValue = *CG_local->getData();
#endif
      return;
   }
   if (name == "neighbors") {
      GenericService< int >* CG_local = dynamic_cast<GenericService< int >*>(service);
      if (CG_local == 0) {
         throw SyntaxErrorException("Expected a int service for neighbors");
      }
#if defined(HAVE_GPU) 
 #if DATAMEMBER_ARRAY_ALLOCATION == OPTION_3
      _container->um_neighbors[index].insert(CG_local->getData());
 #elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_4
      _container->um_neighbors_num_elements[index] +=1;
      auto um_neighbors_index = _container->um_neighbors_start_offset[index] + _container->um_neighbors_num_elements[index]-1;
      _container->um_neighbors[um_neighbors_index] = (CG_local->getData());
 #elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_4b
      _container->um_neighbors_num_elements[index] +=1;
      auto um_neighbors_index = index * _container->um_neighbors_max_elements + _container->um_neighbors_num_elements[index]-1;
      _container->um_neighbors[um_neighbors_index] = (CG_local->getData());
 #elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_5
      assert(0);
 #endif
#else
      neighbors.insert(CG_local->getData());
#endif
      return;
   }
   throw SyntaxErrorException(name + " is not an acceptable service");
}

const CG_LifeNodeSharedMembers& CG_LifeNode::getSharedMembers() const
{
   return *CG_LifeNodeCompCategory::CG_sharedMembers;
}

CG_LifeNodeSharedMembers& CG_LifeNode::getNonConstSharedMembers() 
{
   return *CG_LifeNodeCompCategory::CG_sharedMembers;
}

void CG_LifeNode::addPostVariable(VariableDescriptor* CG_variable, ParameterSet* CG_pset) 
{
   CG_LifeNodeOutAttrPSet* CG_castedPSet = dynamic_cast <CG_LifeNodeOutAttrPSet*>(CG_pset);
   bool noPredicateMatch= true; 
   bool matchPredicateAndCast= false; 
   checkAndAddPostVariable(CG_variable);
}

void CG_LifeNode::addPostEdge(Edge* CG_edge, ParameterSet* CG_pset) 
{
   CG_LifeNodeOutAttrPSet* CG_castedPSet = dynamic_cast <CG_LifeNodeOutAttrPSet*>(CG_pset);
   bool noPredicateMatch= true; 
   bool matchPredicateAndCast= false; 
   checkAndAddPostEdge(CG_edge);
}

void CG_LifeNode::addPostNode(NodeDescriptor* CG_node, ParameterSet* CG_pset) 
{
   CG_LifeNodeOutAttrPSet* CG_castedPSet = dynamic_cast <CG_LifeNodeOutAttrPSet*>(CG_pset);
   bool noPredicateMatch= true; 
   bool matchPredicateAndCast= false; 
   checkAndAddPostNode(CG_node);
}

void CG_LifeNode::addPreConstant(Constant* CG_constant, ParameterSet* CG_pset) 
{
   CG_LifeNodeInAttrPSet* CG_castedPSet = dynamic_cast <CG_LifeNodeInAttrPSet*>(CG_pset);
   bool noPredicateMatch= true; 
   bool matchPredicateAndCast= false; 
   checkAndAddPreConstant(CG_constant);
}

void CG_LifeNode::addPreVariable(VariableDescriptor* CG_variable, ParameterSet* CG_pset) 
{
   CG_LifeNodeInAttrPSet* CG_castedPSet = dynamic_cast <CG_LifeNodeInAttrPSet*>(CG_pset);
   bool noPredicateMatch= true; 
   bool matchPredicateAndCast= false; 
   checkAndAddPreVariable(CG_variable);
}

void CG_LifeNode::addPreEdge(Edge* CG_edge, ParameterSet* CG_pset) 
{
   CG_LifeNodeInAttrPSet* CG_castedPSet = dynamic_cast <CG_LifeNodeInAttrPSet*>(CG_pset);
   bool noPredicateMatch= true; 
   bool matchPredicateAndCast= false; 
   checkAndAddPreEdge(CG_edge);
}

//#if defined(HAVE_GPU) 
//bool CG_LifeNode::addPreNode(NodeDescriptor* CG_node, ParameterSet* CG_pset) 
//#else
void CG_LifeNode::addPreNode(NodeDescriptor* CG_node, ParameterSet* CG_pset) 
//#endif
{
   /* TODO - here it requires ShallowArray to be the same as ShallowArray_Flat */
   ValueProducer* CG_ValueProducerPtr = dynamic_cast<ValueProducer*>(CG_node->getNode());
   CG_LifeNodeInAttrPSet* CG_castedPSet = dynamic_cast <CG_LifeNodeInAttrPSet*>(CG_pset);
   bool noPredicateMatch= true; 
   bool matchPredicateAndCast= false; 
   bool castMatchLocal = true;
   noPredicateMatch = true;
   if (CG_ValueProducerPtr == 0) {
#if !defined(NOWARNING_DYNAMICCAST) 
      std::cerr << "Dynamic Cast of ValueProducer failed in LifeNode" << std::endl;
#endif
      castMatchLocal = false;
   }

   if (castMatchLocal) { 
      if (matchPredicateAndCast) {
         std::cerr << "WARNING: You already have a cast match of predicate" << R"()";
         assert(0);
      }; 
      matchPredicateAndCast = true; 
#if defined(HAVE_GPU) 
 #if DATAMEMBER_ARRAY_ALLOCATION == OPTION_3
      _container->um_neighbors[index].insert(CG_ValueProducerPtr->CG_get_ValueProducer_value());
 #elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_4
      _container->um_neighbors_num_elements[index] +=1;
      auto um_neighbors_index = _container->um_neighbors_start_offset[index] + _container->um_neighbors_num_elements[index]-1;
      _container->um_neighbors[um_neighbors_index] = (CG_ValueProducerPtr->CG_get_ValueProducer_value());
 #elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_4b
      //TUAN TODO see if we can improve speed of this
      _container->um_neighbors_num_elements[index] +=1;
      auto um_neighbors_index = index * _container->um_neighbors_max_elements + _container->um_neighbors_num_elements[index]-1;
      _container->um_neighbors[um_neighbors_index] = (CG_ValueProducerPtr->CG_get_ValueProducer_value());
 #elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_5
      assert(0);
 #endif
#else
      neighbors.insert(CG_ValueProducerPtr->CG_get_ValueProducer_value());
#endif
   } 

   checkAndAddPreNode(CG_node);
   assert(noPredicateMatch || matchPredicateAndCast);
//#if defined(HAVE_GPU) 
//   return castMatchLocal;
//#endif
}

ConnectionIncrement* CG_LifeNode::getComputeCost() const
{
#if 0
   return &_computeCost;
#endif
   return NULL;
}

CG_LifeNode::CG_LifeNode() 
   : ValueProducer(), NodeBase()
#if ! (defined(HAVE_GPU) )
     , value(0), publicValue(0)
#endif
{
   //TUAN make sure _container is pointed to the right one
}

CG_LifeNode::~CG_LifeNode() 
{
}

#if defined(HAVE_GPU) 
CG_LifeNodeCompCategory* CG_LifeNode::_container=nullptr; //instantiation 
#endif
