// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BMC-YKT-08-23-2011-2
//
// (C) Copyright IBM Corp. 2005-2014  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

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

void StructType::getInstance(std::auto_ptr<DataItem> & adi, 
			     std::vector<DataItem*> const * args, 
			     LensContext* c)
{
   StructDataItem* sdi = new StructDataItem;

   std::auto_ptr<Struct> as;
   getStruct(as);
   as->initialize(c, *args);
   sdi->setStruct(as);
   adi.reset(sdi);
}

void StructType::getInstance(std::auto_ptr<DataItem> & adi, 
			     const NDPairList& ndplist, 
			     LensContext* c)
{
   StructDataItem* sdi = new StructDataItem;

   std::auto_ptr<Struct> as;
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
      std::auto_ptr<Struct> dup;
      (*it)->duplicate(dup);
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
