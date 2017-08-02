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

#include "PrintFunctor.h"
#include "FunctorType.h"
#include "FunctorDataItem.h"
#include "LensContext.h"
#include "InstanceFactoryQueriable.h"
//#include <iostream>
#include "DataItem.h"
#include "FunctorDataItem.h"
#include "DataItemQueriable.h"
#include "Functor.h"

void PrintFunctor::doInitialize(LensContext *c, 
				const std::vector<DataItem*>& args)
{
}


void PrintFunctor::doExecute(LensContext *c, 
			     const std::vector<DataItem*>& args, 
			     std::auto_ptr<DataItem>& rvalue)
{
   std::vector<DataItem*>::const_iterator i, 
      begin = args.begin(), end = args.end();
   for (i = begin;i!=end;++i)
      std::cout << (*i)->getString() <<std::endl;

}


void PrintFunctor::duplicate(std::auto_ptr<Functor> &fap) const
{
   fap.reset(new PrintFunctor(*this));
}


PrintFunctor::PrintFunctor()
{
}

PrintFunctor::~PrintFunctor()
{
}
