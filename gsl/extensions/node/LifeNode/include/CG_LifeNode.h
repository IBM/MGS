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

#ifndef CG_LifeNode_H
#define CG_LifeNode_H

#include "Lens.h"
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

class CG_LifeNodeCompCategory;
class ConnectionIncrement;
class Constant;
class Edge;
class Node;
class Variable;
class VariableDescriptor;

class CG_LifeNode : public ValueProducer, public NodeBase
{
//#if defined(HAVE_GPU) && defined(__NVCC__)
//#define um_value (_container->um_value[index]) 
//#define um_publicValue (_container->um_publicValue[index]) 
//#define um_neighbors (_container->um_neighbors[index]) 
//#endif
   friend class CG_LifeNodePublisher;
#ifdef HAVE_MPI
   friend class CG_LifeNodeCompCategory;
#endif
   friend class LifeNodeCompCategory;
   public:
      virtual int* CG_get_ValueProducer_value();
      virtual const char* getServiceName(void* data) const;
      virtual const char* getServiceDescription(void* data) const;
      virtual Publisher* getPublisher();
      virtual void initialize(ParameterSet* CG_initPSet);
      virtual CUDA_CALLABLE void initialize(RNG& rng)=0;
      virtual CUDA_CALLABLE void update(RNG& rng)=0;
      virtual CUDA_CALLABLE void copy(RNG& rng)=0;
      virtual void getInitializationParameterSet(std::unique_ptr<ParameterSet>& initPSet) const;
      virtual void getInAttrParameterSet(std::unique_ptr<ParameterSet>& CG_castedPSet) const;
      virtual void getOutAttrParameterSet(std::unique_ptr<ParameterSet>& CG_castedPSet) const;
      virtual void acceptService(Service* service, const std::string& name);
      const CG_LifeNodeSharedMembers& getSharedMembers() const;
      virtual void addPostVariable(VariableDescriptor* CG_variable, ParameterSet* CG_pset);
      virtual void addPostEdge(Edge* CG_edge, ParameterSet* CG_pset);
      virtual void addPostNode(NodeDescriptor* CG_node, ParameterSet* CG_pset);
      virtual void addPreConstant(Constant* CG_constant, ParameterSet* CG_pset);
      virtual void addPreVariable(VariableDescriptor* CG_variable, ParameterSet* CG_pset);
      virtual void addPreEdge(Edge* CG_edge, ParameterSet* CG_pset);
      virtual void addPreNode(NodeDescriptor* CG_node, ParameterSet* CG_pset);
      virtual ConnectionIncrement* getComputeCost() const;
      CG_LifeNode();
      virtual ~CG_LifeNode();
   protected:
#ifdef HAVE_MPI
      void CG_send_copy(OutputStream* stream) const
      {
         MarshallerInstance<int > mi0;
         //mi0.marshall(stream, publicValue);
         //TODO revise here
         //mi0.marshall(stream, _container->um_publicValue[index]);
      }
#endif
#ifdef HAVE_MPI
      void CG_getSendType_copy(std::vector<int>& blengths, std::vector<MPI_Aint>& blocs) const;
#endif
#ifdef HAVE_MPI
      void CG_send_FLUSH_LENS(OutputStream* stream) const;
#endif
#ifdef HAVE_MPI
      void CG_getSendType_FLUSH_LENS(std::vector<int>& blengths, std::vector<MPI_Aint>& blocs) const;
#endif
      virtual TriggerableBase::EventType createTriggerableCaller(const std::string& CG_triggerableFunctionName, NDPairList* CG_triggerableNdpList, std::unique_ptr<TriggerableCaller>& CG_triggerableCaller);

#if defined(HAVE_GPU) && defined(__NVCC__)
      int index; //the index in the array
   public:
      //TUAN TODO: write a method function to provide the access to '_container' 
      //    so that we don't make '_container' public
      // and also to avoid the error 'access incomplete type' if we make '_container' as private
      //
      CG_LifeNodeCompCategory* _container;
      //LifeNodeCompCategory* _container;
   protected:
      //int& value(){ return _container->um_value[index];}
      //inline int& publicValue(){ return _container->um_publicValue[index];}
      //inline ShallowArray_Flat<int*> & neighbors(){ return _container->um_neighbors[index];}
      void setCompCategory(int _index, CG_LifeNodeCompCategory* cg) { index = _index; _container=cg; }
#else
      int value;
      int publicValue;
      ShallowArray< int* > neighbors;
#endif
   private:
      CG_LifeNodeSharedMembers& getNonConstSharedMembers();
};

#endif
