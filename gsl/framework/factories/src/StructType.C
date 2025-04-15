// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "StructType.h"
#include "StructDataItem.h"
#include "Struct.h"
#include "DataItemQueriable.h"
#include "InstanceFactoryQueriable.h"
#include "NDPairList.h"

StructType::StructType()
   : InstanceFactory()
{
}

StructType::StructType(const StructType& rv)
   : InstanceFactory(rv)
{
   copyContents(rv);
}

StructType& StructType::operator=(const StructType& rv)
{
   if (this == &rv) {
      return *this;
   }
   InstanceFactory::operator=(rv);
   destructContents();
   copyContents(rv);
   return *this;
}

void StructType::getInstance(std::unique_ptr<DataItem> & adi, 
			     std::vector<DataItem*> const * args, 
			     LensContext* c)
{
   StructDataItem* sdi = new StructDataItem;

   std::unique_ptr<Struct> as;
   getStruct(as);
   as->initialize(c, *args);
   sdi->setStruct(as);
   adi.reset(sdi);
}

void StructType::getInstance(std::unique_ptr<DataItem> & adi, 
			     const NDPairList& ndplist, 
			     LensContext* c)
{
   StructDataItem* sdi = new StructDataItem;

   std::unique_ptr<Struct> as;
   getStruct(as);
   as->initialize(ndplist);
   sdi->setStruct(as);
   adi.reset(sdi);
}

StructType::~StructType()
{
   destructContents();
}

void StructType::copyContents(const StructType& rv)
{
   std::list<Struct*>::const_iterator it, end = rv._structList.end();
   for (it = rv._structList.begin(); it!=end; ++it) {
      std::unique_ptr<Struct> dup;
      (*it)->duplicate(std::move(dup));
      _structList.push_back(dup.release());
   }
}

void StructType::destructContents()
{
   std::list<Struct*>::iterator it, end = _structList.end();
   for (it = _structList.begin(); it!=end; ++it) {
      delete (*it);
   }
}
