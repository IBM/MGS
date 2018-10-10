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
      void addPublisher(std::unique_ptr<Publisher> & ptrPublisher);
      void removePublisher(std::string publisherName);
      ~PublisherRegistry();
   private:
      Simulation& _sim;
      std::list<Publisher*> _pubList;
};
#endif
