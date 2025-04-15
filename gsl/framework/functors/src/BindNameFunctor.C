// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "BindNameFunctor.h"
#include "FunctorType.h"
#include "CustomStringDataItem.h"
#include "LensContext.h"
//#include <iostream>
//#include <sstream>
#include "DataItem.h"
#include "NumericDataItem.h"
#include "NumericArrayDataItem.h"
#include "FloatDataItem.h"
#include "IntDataItem.h"
#include "FunctorDataItem.h"
#include "NDPairListDataItem.h"
#include "InstanceFactoryQueriable.h"
#include "NDPair.h"
#include "DataItemQueriable.h"
#include "FunctorDataItem.h"
#include "SyntaxErrorException.h"
#include <memory>

void BindNameFunctor::doInitialize(LensContext *c, 
				   const std::vector<DataItem*>& args)
{
   std::vector<DataItem*>::const_iterator iter, 
      begin = args.begin(), end = args.end();

   std::unique_ptr<DataItem> di;
   for ( iter = begin; iter != end; ++iter) {
      std::pair<std::string, DataItem*> curElem;

      // get the name
      CustomStringDataItem* sdi;
     sdi = dynamic_cast<CustomStringDataItem*>(*iter);
      if (sdi == 0) {
	 throw SyntaxErrorException(
	    "Dynamic cast of DataItem to CustomStringDataItem failed in BindNameFunctor");
      }
      curElem.first = sdi->getString();

      // get the value
      iter++;
      (*iter)->duplicate(di);
      curElem.second = di.release();
      _nameDataItems.push_back(curElem);
   }
}


void BindNameFunctor::doExecute(LensContext *c, 
				const std::vector<DataItem*>& args, 
				std::unique_ptr<DataItem>& rvalue)
{
   // Get a list of names and values and store them in a NDPairList,
   // then reset the value

   std::vector<NDPairGenerator>::iterator iter, begin, end;

   std::unique_ptr<NDPairList > ndpList(new NDPairList);
   begin = _nameDataItems.begin();
   end = _nameDataItems.end();
 
   std::unique_ptr<DataItem> di;
   for ( iter = begin; iter != end; ++iter) {
      FunctorDataItem* fdi = dynamic_cast<FunctorDataItem*>(iter->second);
      if (fdi) {
	 std::vector<DataItem*> nullArgs;
	 std::unique_ptr<DataItem> rval_ap;
	 fdi->getFunctor()->execute(c, nullArgs, rval_ap);
         NumericDataItem* ndi = dynamic_cast<NumericDataItem*>(rval_ap.get());
         NumericArrayDataItem* nadi = dynamic_cast<NumericArrayDataItem*>(rval_ap.get());
         if (ndi ==0 && nadi ==0) {
	    throw SyntaxErrorException(
	       "Functor doesn't return a numeric or numeric array value in BindNameFunctor");
         }
	 if (ndi) ndi->duplicate(di);
	 else if (nadi) nadi->duplicate(di);
      } else {
	 iter->second->duplicate(di);
      }
      NDPair* ndp = new NDPair(iter->first, di);
      ndpList->push_back(ndp);
   }

   NDPairListDataItem *nv_di = new NDPairListDataItem;
   nv_di->setNDPairList(ndpList);
   rvalue.reset(nv_di);
}


void BindNameFunctor::duplicate(std::unique_ptr<Functor>&& fap) const
{
   fap=std::make_unique<BindNameFunctor>(*this);
}


BindNameFunctor::BindNameFunctor()
{
}


BindNameFunctor::BindNameFunctor(const BindNameFunctor &f)
{

   std::unique_ptr<DataItem> di;
   std::vector<NDPairGenerator>::const_iterator it, 
      end = f._nameDataItems.end();
   for (it = f._nameDataItems.begin(); it != end; ++it) {
      it->second->duplicate(di);
      _nameDataItems.push_back(NDPairGenerator(it->first, di.release()));
   }
}


BindNameFunctor::~BindNameFunctor()
{
   std::vector<NDPairGenerator>::iterator it, end = _nameDataItems.end();
   for (it = _nameDataItems.begin(); it != end; ++it) {
      delete it->second;
   }
}
