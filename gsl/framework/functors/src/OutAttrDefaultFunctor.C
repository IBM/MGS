// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BCM-YKT-11-19-2015
//
// (C) Copyright IBM Corp. 2005-2015  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

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
      csf._pset->duplicate(_pset);
   }
}

void OutAttrDefaultFunctor::duplicate (std::auto_ptr<Functor> &fap) const
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
   std::auto_ptr<DataItem>& rvalue)
{
   ConnectionContext *cc = c->connectionContext;
   std::auto_ptr<ParameterSet> pset;
   NodeType *nt = cc->sourceNode->getGridLayerDescriptor()->getNodeType();
   nt->getOutAttrParameterSet(pset);

   ParameterSetDataItem *psdi= new ParameterSetDataItem();
   psdi->setParameterSet(pset);
   rvalue.reset(psdi);
}
