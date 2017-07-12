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

#include "InstanceFactory.h"
#include "LensContext.h"
#include "DataItem.h"

InstanceFactory::InstanceFactory()
{
}

InstanceFactory::InstanceFactory(const InstanceFactory& rv)
{
   copyOwnedHeap(rv);
}

InstanceFactory& InstanceFactory::operator=(const InstanceFactory& rv)
{
   if (this != &rv) {
      destructOwnedHeap();
      copyOwnedHeap(rv);
   }
   return *this;
}

InstanceFactory::~InstanceFactory()
{
   destructOwnedHeap();
}

void InstanceFactory::copyOwnedHeap(const InstanceFactory& rv)
{
   std::vector<std::vector<std::pair<std::string, DataItem*> > >::const_iterator it
      , end = rv._parameterDescription.end();
   for(it = rv._parameterDescription.begin(); it != end; ++it) {
      std::vector<std::pair<std::string, DataItem*> >::const_iterator it2
	 , end2 = (*it).end();
      std::vector<std::pair<std::string, DataItem*> > nVec;
      for (it2 = (*it).begin(); it2 != end2; ++it2) {
	 std::auto_ptr<DataItem> dup;
	 ((*it2).second)->duplicate(dup);
	 std::pair<std::string, DataItem*> nPair;
	 nPair.first = (*it2).first;
	 nPair.second = dup.release();
	 nVec.push_back(nPair);
      }
      _parameterDescription.push_back(nVec);
   }
   std::vector<DataItem*>::const_iterator it2, end2 = rv._instances.end();
   for(it2 = rv._instances.begin(); it2!=end2; ++it2) {
      std::auto_ptr<DataItem> dup;
      (*it2)->duplicate(dup);
      _instances.push_back(dup.release());
   }
}

void InstanceFactory::destructOwnedHeap()
{
   std::vector<std::vector<std::pair<std::string, DataItem*> > >::iterator it
      , end = _parameterDescription.end();
   for(it = _parameterDescription.begin(); it != end; ++it) {
      std::vector<std::pair<std::string, DataItem*> >::iterator it2
	 , end2 = (*it).end();
      for (it2 = (*it).begin(); it2 != end2; ++it2) {
	 delete (*it2).second;
      }
   }
   std::vector<DataItem*>::iterator it2, end2 = _instances.end();
   for(it2 = _instances.begin(); it2!=end2; ++it2) {
      delete (*it2);
   }
}
