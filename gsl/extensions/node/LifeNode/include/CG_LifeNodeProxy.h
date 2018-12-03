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

#ifndef CG_LifeNodeProxy_H
#define CG_LifeNodeProxy_H

#include "Lens.h"
#if defined(HAVE_MPI)
#include "CG_LifeNodeOutAttrPSet.h"
#include "CG_LifeNodeProxyDemarshaller.h"
#include "CG_LifeNodeSharedMembers.h"
#include "DataItem.h"
#include "DataItemArrayDataItem.h"
#include "DemarshallerInstance.h"
#include "IntArrayDataItem.h"
#include "IntDataItem.h"
#include "NodeProxyBase.h"
#include "ShallowArray.h"
#include "String.h"
#include "ValueProducer.h"
#include <cassert>
#include <iostream>
#include <memory>

class CG_LifeNodeCompCategory;
class Constant;
class Edge;
class Node;
class Publisher;
class Variable;

class CG_LifeNodeProxy : public ValueProducer, public NodeProxyBase
{
   friend class CG_LifeNodePublisher;
   friend class CG_LifeNodeCompCategory;
   friend class LifeNodeCompCategory;
   private:

   class PhaseDemarshaller_FLUSH_LENS : public CG_LifeNodeProxyDemarshaller
   {
   public:
      PhaseDemarshaller_FLUSH_LENS()
      : CG_LifeNodeProxyDemarshaller(), publicValueDemarshaller()      {
   _demarshallers.push_back(&publicValueDemarshaller);
      }
      PhaseDemarshaller_FLUSH_LENS(CG_LifeNodeProxy* proxy);
      void setDestination(CG_LifeNodeProxy *proxy);
      virtual ~PhaseDemarshaller_FLUSH_LENS()
      {
      }
   private:
      DemarshallerInstance< int > publicValueDemarshaller;
   };



   class PhaseDemarshaller_copy : public CG_LifeNodeProxyDemarshaller
   {
   public:
      PhaseDemarshaller_copy()
      : CG_LifeNodeProxyDemarshaller(), publicValueDemarshaller()      {
         _demarshallers.push_back(&publicValueDemarshaller);
      }
      PhaseDemarshaller_copy(CG_LifeNodeProxy* proxy);
      void setDestination(CG_LifeNodeProxy *proxy);
      virtual ~PhaseDemarshaller_copy()
      {
      }
   private:
      DemarshallerInstance< int > publicValueDemarshaller;
   };


   public:
#if defined(HAVE_GPU)
#if PROXY_ALLOCATION == OPTION_3
      void setCompCategory(int _index, CG_LifeNodeCompCategory* __container, int _demarshaller_index)
      {
      index = _index;
       _container = __container;
      demarshaller_index = _demarshaller_index;
      }
#endif
#endif
#if defined(HAVE_GPU)
#if PROXY_ALLOCATION == OPTION_4
      void setCompCategory(int _index, CG_LifeNodeCompCategory* __container)
      {
      index = _index; _container = __container;
      }
#endif
#endif
#if defined(HAVE_GPU)
      CG_LifeNodeCompCategory* getCompCategory()
      {
      return _container;
      }
#endif
#if defined(HAVE_GPU)
#if PROXY_ALLOCATION == OPTION_3
      int getDemarshallerIndex()
      {
      return demarshaller_index;
      }
#endif
#endif
#if defined(HAVE_GPU)
      int getDataIndex()
      {
      return index;
      }
#endif
      virtual int* CG_get_ValueProducer_value();
#if defined(HAVE_MPI)
      static void CG_recv_copy_demarshaller(std::unique_ptr<CG_LifeNodeProxyDemarshaller> &ap);
#endif
#if defined(HAVE_MPI)
      static void CG_recv_FLUSH_LENS_demarshaller(std::unique_ptr<CG_LifeNodeProxyDemarshaller> &ap);
#endif
      const CG_LifeNodeSharedMembers& getSharedMembers();
      virtual void addPostVariable(VariableDescriptor* CG_variable, ParameterSet* CG_pset);
      virtual void addPostEdge(Edge* CG_edge, ParameterSet* CG_pset);
      virtual void addPostNode(NodeDescriptor* CG_node, ParameterSet* CG_pset);
      CG_LifeNodeProxy();
      virtual ~CG_LifeNodeProxy();
   protected:
#ifdef HAVE_GPU
#if defined(HAVE_GPU)
      int index;
#endif
#if defined(HAVE_GPU)
      static CG_LifeNodeCompCategory* _container;
#endif
#if defined(HAVE_GPU)
#if PROXY_ALLOCATION == OPTION_3
      int demarshaller_index;
#endif
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
#endif
