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

#if defined(HAVE_MPI)
#include "Lens.h"
#include "CG_LifeNodeProxy.h"
#include "CG_LifeNodeCompCategory.h"
#include "Constant.h"
#include "Edge.h"
#include "Node.h"
#include "Publisher.h"
#include "Variable.h"
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

CG_LifeNodeProxy::PhaseDemarshaller_FLUSH_LENS::PhaseDemarshaller_FLUSH_LENS(CG_LifeNodeProxy* proxy) 
   : CG_LifeNodeProxyDemarshaller(proxy)
#ifdef HAVE_GPU
#if PROXY_ALLOCATION == OPTION_3
, publicValueDemarshaller(&(((proxy->getCompCategory())->getDemarshaller(proxy->demarshaller_index))->um_publicValue[proxy->index]))
#elif PROXY_ALLOCATION == OPTION_4
   , publicValueDemarshaller(&(proxy->getCompCategory()->proxy_um_publicValue[proxy->getDataIndex()]))
#endif
#else
, publicValueDemarshaller(&(proxy->publicValue))
#endif
{
   _demarshallers.push_back(&publicValueDemarshaller);
}

void CG_LifeNodeProxy::PhaseDemarshaller_FLUSH_LENS::setDestination(CG_LifeNodeProxy *proxy) 
{
   _proxy = proxy;
   #ifdef HAVE_GPU
   #if PROXY_ALLOCATION == OPTION_3
   publicValueDemarshaller.setDestination(&(_proxy->getCompCategory()->getDemarshaller(proxy->getDemarshallerIndex())->um_publicValue[proxy->getDataIndex()]));
   #elif PROXY_ALLOCATION == OPTION_4
publicValueDemarshaller.setDestination(&(_proxy->getCompCategory()->proxy_um_publicValue[proxy->getDataIndex()]));
   #endif
   #else
   publicValueDemarshaller.setDestination(&(_proxy->publicValue));
   #endif
   reset();
}

CG_LifeNodeProxy::PhaseDemarshaller_copy::PhaseDemarshaller_copy(CG_LifeNodeProxy* proxy) 
   : CG_LifeNodeProxyDemarshaller(proxy)
#ifdef HAVE_GPU
#if PROXY_ALLOCATION == OPTION_3
, publicValueDemarshaller(&(((proxy->getCompCategory())->getDemarshallerIndex(proxy->demarshaller_index))->um_publicValue[proxy->index]))
#elif PROXY_ALLOCATION == OPTION_4
   , publicValueDemarshaller(&(proxy->getCompCategory()->proxy_um_publicValue[proxy->getDataIndex()]))
#endif
#else
, publicValueDemarshaller(&(proxy->publicValue))
#endif
{
         _demarshallers.push_back(&publicValueDemarshaller);
}

void CG_LifeNodeProxy::PhaseDemarshaller_copy::setDestination(CG_LifeNodeProxy *proxy) 
{
   _proxy = proxy;
   #ifdef HAVE_GPU
   #if PROXY_ALLOCATION == OPTION_3
   publicValueDemarshaller.setDestination(&(_proxy->getCompCategory()->getDemarshaller(proxy->getDemarshallerIndex())->um_publicValue[proxy->getDataIndex()]));
   #elif PROXY_ALLOCATION == OPTION_4
publicValueDemarshaller.setDestination(&(_proxy->getCompCategory()->proxy_um_publicValue[proxy->getDataIndex()]));
   #endif
   #else
   publicValueDemarshaller.setDestination(&(_proxy->publicValue));
   #endif
         reset();
}

int* CG_LifeNodeProxy::CG_get_ValueProducer_value() 
{
#ifdef HAVE_GPU
#if PROXY_ALLOCATION == OPTION_3
   return &((_container->getDemarshaller(demarshaller_index))->um_publicValue[index]);
#elif PROXY_ALLOCATION == OPTION_4
   return &(_container->proxy_um_publicValue[index]);
#endif

#else
   return &publicValue;
#endif
}

#if defined(HAVE_MPI)
void CG_LifeNodeProxy::CG_recv_copy_demarshaller(std::unique_ptr<CG_LifeNodeProxyDemarshaller> &ap) 
{
   PhaseDemarshaller_copy* di = new PhaseDemarshaller_copy();
   ap.reset(di);
}
#endif

#if defined(HAVE_MPI)
void CG_LifeNodeProxy::CG_recv_FLUSH_LENS_demarshaller(std::unique_ptr<CG_LifeNodeProxyDemarshaller> &ap) 
{
   PhaseDemarshaller_FLUSH_LENS *di = new PhaseDemarshaller_FLUSH_LENS();
   ap.reset(di);
}
#endif

const CG_LifeNodeSharedMembers& CG_LifeNodeProxy::getSharedMembers() 
{
   return *CG_LifeNodeCompCategory::CG_sharedMembers;
}

CG_LifeNodeSharedMembers& CG_LifeNodeProxy::getNonConstSharedMembers() 
{
   return *CG_LifeNodeCompCategory::CG_sharedMembers;
}

void CG_LifeNodeProxy::addPostVariable(VariableDescriptor* CG_variable, ParameterSet* CG_pset) 
{
   CG_LifeNodeOutAttrPSet* CG_castedPSet = dynamic_cast <CG_LifeNodeOutAttrPSet*>(CG_pset);
   bool noPredicateMatch= true; 
   bool matchPredicateAndCast= false; 
   checkAndAddPostVariable(CG_variable);
}

void CG_LifeNodeProxy::addPostEdge(Edge* CG_edge, ParameterSet* CG_pset) 
{
   CG_LifeNodeOutAttrPSet* CG_castedPSet = dynamic_cast <CG_LifeNodeOutAttrPSet*>(CG_pset);
   bool noPredicateMatch= true; 
   bool matchPredicateAndCast= false; 
   checkAndAddPostEdge(CG_edge);
}

void CG_LifeNodeProxy::addPostNode(NodeDescriptor* CG_node, ParameterSet* CG_pset) 
{
   CG_LifeNodeOutAttrPSet* CG_castedPSet = dynamic_cast <CG_LifeNodeOutAttrPSet*>(CG_pset);
   bool noPredicateMatch= true; 
   bool matchPredicateAndCast= false; 
   checkAndAddPostNode(CG_node);
}

CG_LifeNodeProxy::CG_LifeNodeProxy() 
   : ValueProducer(), NodeProxyBase()
#if ! defined(HAVE_GPU)
   , value(0)
#endif
#if ! defined(HAVE_GPU)
   , publicValue(0)
#endif
{
}

CG_LifeNodeProxy::~CG_LifeNodeProxy() 
{
}

#if defined(HAVE_GPU)
CG_LifeNodeCompCategory* CG_LifeNodeProxy::_container;
#endif
#endif
