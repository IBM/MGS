// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
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
