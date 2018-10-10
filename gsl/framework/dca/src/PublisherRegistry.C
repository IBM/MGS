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

#ifdef HAVE_MPI
#include <mpi.h>
#endif
#include "PublisherRegistry.h"
#include "Publisher.h"
#include "Simulation.h"

#include <sstream>
#include <iostream>

PublisherRegistry::PublisherRegistry(Simulation& s)
: _sim(s)
{
}


const std::list<Publisher*>& PublisherRegistry::getPublisherList()
{
   return _pubList;
}


Publisher* PublisherRegistry::getPublisher(std::string publisherName)
{
   Publisher* rval = 0;
   std::list<Publisher*>::iterator end = _pubList.end();
   for (std::list<Publisher*>::iterator iter = _pubList.begin(); iter != end; iter++) {
      if (publisherName == (*iter)->getName()) {
         rval = *iter;
      }
   }
   if (rval == 0) {
      std::cerr<<"Publisher "<<publisherName<<" not found in PublisherRegistry!"<<std::endl;
      exit(-1);
   }
   return rval;
}


void PublisherRegistry::removePublisher(std::string publisherName)
{
   std::list<Publisher*>::iterator end = _pubList.end();
   for (std::list<Publisher*>::iterator iter = _pubList.begin(); iter != end; iter++) {
      if (publisherName == (*iter)->getName()) {
         Publisher* p = (*iter);
         _pubList.erase(iter);
         delete p;
      }
   }
}


void PublisherRegistry::addPublisher(std::unique_ptr<Publisher> & ptrPublisher)
{
   bool add = true;
   std::list<Publisher*>::iterator end = _pubList.end();
   for (std::list<Publisher*>::iterator iter = _pubList.begin(); iter != end; iter++) {
      if (ptrPublisher->getName() == (*iter)->getName()) {
         add = false;
         break;
      }
   }
   if (add) _pubList.push_back(ptrPublisher.release());
   return;
}


PublisherRegistry::~PublisherRegistry()
{
   std::list<Publisher*>::iterator end = _pubList.end();
   for (std::list<Publisher*>::iterator iter = _pubList.begin(); iter != end; iter++) {
      delete (*iter);
   }
}
