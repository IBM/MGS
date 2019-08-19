// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BCM-YKT-07-18-2017
//
// (C) Copyright IBM Corp. 2005-2017  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#include "InAttrDefaultFunctor.h"
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

InAttrDefaultFunctor::InAttrDefaultFunctor()
{
}

InAttrDefaultFunctor::InAttrDefaultFunctor(const InAttrDefaultFunctor& csf)
{
   if (csf._pset.get()) {
      csf._pset->duplicate(_pset);
   }
}

void InAttrDefaultFunctor::duplicate(std::unique_ptr<Functor> &fap) const
{
   fap.reset(new InAttrDefaultFunctor(*this));
}


InAttrDefaultFunctor::~InAttrDefaultFunctor()
{
}


void InAttrDefaultFunctor::doInitialize(LensContext *c, 
					const std::vector<DataItem*>& args)
{
}


void InAttrDefaultFunctor::doExecute(LensContext *c, 
				     const std::vector<DataItem*>& args, 
				     std::unique_ptr<DataItem>& rvalue)
{
   ConnectionContext *cc = c->connectionContext;
   std::unique_ptr<ParameterSet> pset;

   NodeType *nt = cc->destinationNode->getGridLayerDescriptor()->getNodeType();
   nt->getInAttrParameterSet(pset);

   ParameterSetDataItem *psdi= new ParameterSetDataItem();
   psdi->setParameterSet(pset);
   rvalue.reset(psdi);

}
