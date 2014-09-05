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

#ifndef PUBLISHERREGISTRY_H
#define PUBLISHERREGISTRY_H
#include "Copyright.h"

#include <memory>
#include <list>
#include <string>


class Simulation;
class Publisher;

class PublisherRegistry
{
   friend class PublisherRegistryQueriable;

   public:
      PublisherRegistry(Simulation& s);
      const std::list<Publisher*> & getPublisherList();
      Publisher* getPublisher(std::string publisherName);
      void addPublisher(std::auto_ptr<Publisher> & ptrPublisher);
      void removePublisher(std::string publisherName);
      ~PublisherRegistry();
   private:
      Simulation& _sim;
      std::list<Publisher*> _pubList;
};
#endif
