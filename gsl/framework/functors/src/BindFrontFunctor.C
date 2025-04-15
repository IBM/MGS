// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "BindFrontFunctor.h"
#include "FunctorType.h"
#include "LensContext.h"
//#include <iostream>
#include "DataItem.h"
#include "FunctorDataItem.h"
#include "InstanceFactoryQueriable.h"
#include "DataItemQueriable.h"
#include "FunctorDataItem.h"
#include "SyntaxErrorException.h"

class Functor;
class FunctorType;
class Simulation;

void BindFrontFunctor::doInitialize(LensContext *c,
				    const std::vector<DataItem*>& args)
{

   // get functor and argument list

   FunctorDataItem *fdi = dynamic_cast<FunctorDataItem*>(args[0]);
   if (fdi == 0) {
      throw SyntaxErrorException(
	 "Dynamic cast of DataItem to FunctorDataItem failed in BindFrontFunctor");
   }

   std::unique_ptr<Functor> fap;
   fdi->getFunctor()->duplicate(std::move(fap));
   _bind_functor = fap.release();

   std::vector<DataItem*>::const_iterator iter, 
      begin = args.begin(), end = args.end();
   begin++;

   std::unique_ptr<DataItem> DI;
   for ( iter = begin; iter != end; ++iter ) {
      (*iter)->duplicate(DI);
      _bind_args.push_back( DI.release());
   }
}


void BindFrontFunctor::doExecute(LensContext *c, 
				 const std::vector<DataItem*>& args, 
				 std::unique_ptr<DataItem>& rvalue)
{
   std::vector<DataItem*> expArgs;

   std::vector<DataItem*>::const_iterator beginDI = args.begin(), 
      endDI = args.end();
   std::vector<DataItem*>::const_iterator bindBegin = _bind_args.begin(), 
      bindEnd = _bind_args.end();

   expArgs.insert( expArgs.end(), bindBegin, bindEnd );

   expArgs.insert(expArgs.end(), beginDI, endDI);

   _bind_functor->execute(c, expArgs, rvalue);
}


void BindFrontFunctor::duplicate(std::unique_ptr<Functor>&& fap) const
{
   fap.reset(new BindFrontFunctor(*this));
}


BindFrontFunctor::BindFrontFunctor()
: _bind_functor(0)
{
}


BindFrontFunctor::BindFrontFunctor(const BindFrontFunctor &f)
: _bind_functor(0)
{
   if (f._bind_functor) {
      std::unique_ptr<Functor> fap;
      f._bind_functor->duplicate(std::move(fap));
      _bind_functor = fap.release();
   }

   std::vector<DataItem*>::const_iterator iter, 
      begin = f._bind_args.begin(), end = f._bind_args.end();

   std::unique_ptr<DataItem> DI;
   for ( iter = begin; iter != end; ++iter ) {
      (*iter)->duplicate(DI);
      _bind_args.push_back(DI.release());
   }
}


BindFrontFunctor::~BindFrontFunctor()
{
   std::vector<DataItem*>::iterator iter, begin = _bind_args.begin(), 
      end = _bind_args.end();
   for ( iter = begin; iter != end; ++iter )
      delete *iter;

}
