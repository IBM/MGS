// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "OutAttrDefaultFunctor.h"
#include "LensContext.h"
#include "DataItem.h"
#include "FunctorType.h"
#include "NodeSet.h"
#include "GridLayerDescriptor.h"
#include "Node.h"
#include "ParameterSet.h"
#include "ParameterSetDataItem.h"
#include "NodeType.h"
#include "Simulation.h"
#include "ConnectionContext.h"
#include "NodeInitializerFunctor.h"
#include "InstanceFactoryQueriable.h"
#include "DataItemQueriable.h"
#include "FunctorDataItem.h"

OutAttrDefaultFunctor::OutAttrDefaultFunctor()
{
}

OutAttrDefaultFunctor::OutAttrDefaultFunctor(const OutAttrDefaultFunctor& csf)
{
   if (csf._pset.get()) {
      csf._pset->duplicate(std::move(_pset));
   }
}

void OutAttrDefaultFunctor::duplicate (std::unique_ptr<Functor>&& fap) const
{
   fap.reset(new OutAttrDefaultFunctor(*this));
}


OutAttrDefaultFunctor::~OutAttrDefaultFunctor()
{
}


void OutAttrDefaultFunctor::doInitialize(LensContext *c, 
					 const std::vector<DataItem*>& args)
{
}


void OutAttrDefaultFunctor::doExecute(
   LensContext *c, const std::vector<DataItem*>& args, 
   std::unique_ptr<DataItem>& rvalue)
{
   ConnectionContext *cc = c->connectionContext;
   std::unique_ptr<ParameterSet> pset;
   NodeType *nt = cc->sourceNode->getGridLayerDescriptor()->getNodeType();
   nt->getOutAttrParameterSet(std::move(pset));

   ParameterSetDataItem *psdi= new ParameterSetDataItem();
   psdi->setParameterSet(pset);
   rvalue.reset(psdi);
}
