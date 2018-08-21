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

#include "ReverseFunctor.h"
#include "CG_ReverseFunctorBase.h"
#include "LensContext.h"
#include "ConnectionContext.h"
#include "NodeDescriptor.h"
#include <memory>

void ReverseFunctor::userInitialize(LensContext* CG_c, Functor*& f) 
{
}

void ReverseFunctor::userExecute(LensContext* CG_c) 
{
   std::vector<DataItem*> nullArgs;
   std::auto_ptr<DataItem> rval_ap;

   init.f->execute(CG_c, nullArgs, rval_ap);
   ConnectionContext *cc = CG_c->connectionContext;
   
   NodeDescriptor* sn = cc->sourceNode;
   cc->sourceNode = cc->destinationNode;
   cc->destinationNode = sn;

   sn = cc->sourceRefNode;
   cc->sourceRefNode = cc->destinationRefNode;
   cc->destinationRefNode = sn;
}

ReverseFunctor::ReverseFunctor() 
   : CG_ReverseFunctorBase()
{
}

ReverseFunctor::~ReverseFunctor() 
{
}

void ReverseFunctor::duplicate(std::auto_ptr<ReverseFunctor>& dup) const
{
   dup.reset(new ReverseFunctor(*this));
}

void ReverseFunctor::duplicate(std::auto_ptr<Functor>& dup) const
{
   dup.reset(new ReverseFunctor(*this));
}

void ReverseFunctor::duplicate(std::auto_ptr<CG_ReverseFunctorBase>& dup) const
{
   dup.reset(new ReverseFunctor(*this));
}

