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

#ifndef CG_LifeNode_H
#define CG_LifeNode_H

#include "Lens.h"
#include "CG_LifeNodeInAttrPSet.h"
#include "CG_LifeNodeOutAttrPSet.h"
#include "CG_LifeNodePSet.h"
#if defined(HAVE_MPI)
#include "CG_LifeNodeProxyDemarshaller.h"
#endif
#include "CG_LifeNodePublisher.h"
#include "CG_LifeNodeSharedMembers.h"
#include "DataItem.h"
#include "DataItemArrayDataItem.h"
#include "IntArrayDataItem.h"
#include "IntDataItem.h"
#if defined(HAVE_MPI)
#include "Marshall.h"
#endif
#include "NodeBase.h"
#if defined(HAVE_MPI)
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
   friend class CG_LifeNodePublisher;
#if defined(HAVE_MPI)
   friend class CG_LifeNodeCompCategory;
#endif
   friend class LifeNodeCompCategory;
   public:
#if defined(HAVE_GPU)
      void setCompCategory(int _index, CG_LifeNodeCompCategory* __container)
      {
      index = _index; _container = __container;
      }
#endif
#if defined(HAVE_GPU)
      CG_LifeNodeCompCategory* getCompCategory()
      {
      return _container;
      }
#endif
      virtual int* CG_get_ValueProducer_value();
      virtual const char* getServiceName(void* data) const;
      virtual const char* getServiceDescription(void* data) const;
      virtual Publisher* getPublisher();
      virtual void initialize(ParameterSet* CG_initPSet);
      virtual void initialize(RNG& rng)=0;
      virtual void update(RNG& rng)=0;
      virtual void copy(RNG& rng)=0;
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
#if defined(HAVE_MPI)
      void CG_send_copy(OutputStream* stream) const
      {
         MarshallerInstance<int > mi0;
         mi0.marshall(stream, publicValue);
      }
#endif
#if defined(HAVE_MPI)
      void CG_getSendType_copy(std::vector<int>& blengths, std::vector<MPI_Aint>& blocs) const;
#endif
#if defined(HAVE_MPI)
      void CG_send_FLUSH_LENS(OutputStream* stream) const;
#endif
#if defined(HAVE_MPI)
      void CG_getSendType_FLUSH_LENS(std::vector<int>& blengths, std::vector<MPI_Aint>& blocs) const;
#endif
      virtual TriggerableBase::EventType createTriggerableCaller(const std::string& CG_triggerableFunctionName, NDPairList* CG_triggerableNdpList, std::unique_ptr<TriggerableCaller>& CG_triggerableCaller);
#ifdef HAVE_GPU
#if defined(HAVE_GPU)
      int index;
#endif
#if defined(HAVE_GPU)
      static CG_LifeNodeCompCategory* _container;
#endif
#else
#if ! defined(HAVE_GPU)
      int value;
#endif
#if ! defined(HAVE_GPU)
      int publicValue;
#endif
#if ! defined(HAVE_GPU)
      ShallowArray< int* > neighbors;
#endif
#endif
   private:
      CG_LifeNodeSharedMembers& getNonConstSharedMembers();
};

#endif
