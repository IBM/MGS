// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
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
   std::unique_ptr<DataItem> rval_ap;

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

void ReverseFunctor::duplicate(std::unique_ptr<ReverseFunctor>&& dup) const
{
   dup.reset(new ReverseFunctor(*this));
}

void ReverseFunctor::duplicate(std::unique_ptr<Functor>&& dup) const
{
   dup.reset(new ReverseFunctor(*this));
}

void ReverseFunctor::duplicate(std::unique_ptr<CG_ReverseFunctorBase>&& dup) const
{
   dup.reset(new ReverseFunctor(*this));
}

