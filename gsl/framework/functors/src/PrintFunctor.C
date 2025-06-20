// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "PrintFunctor.h"
#include "FunctorType.h"
#include "FunctorDataItem.h"
#include "GslContext.h"
#include "InstanceFactoryQueriable.h"
//#include <iostream>
#include "DataItem.h"
#include "FunctorDataItem.h"
#include "DataItemQueriable.h"
#include "Functor.h"
#include "Simulation.h"

void PrintFunctor::doInitialize(GslContext *c, 
				const std::vector<DataItem*>& args)
{
}


void PrintFunctor::doExecute(GslContext *c, 
			     const std::vector<DataItem*>& args, 
			     std::unique_ptr<DataItem>& rvalue)
{
   std::vector<DataItem*>::const_iterator i, 
      begin = args.begin(), end = args.end();
   if (c->sim->getRank() == 0)
   {
      for (i = begin;i!=end;++i)
	 std::cout << (*i)->getString() <<std::endl;
   }

}


void PrintFunctor::duplicate(std::unique_ptr<Functor>&& fap) const
{
   fap.reset(new PrintFunctor(*this));
}


PrintFunctor::PrintFunctor()
{
}

PrintFunctor::~PrintFunctor()
{
}
