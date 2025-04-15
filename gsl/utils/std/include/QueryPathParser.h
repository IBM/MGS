// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef QUERYPATHPARSER_H
#define QUERYPATHPARSER_H
#include "Copyright.h"

#include <string>


class Simulation;
class Service;
class Publisher;
class TriggerType;
class Queriable;

class QueryPathParser
{
   public:
      QueryPathParser(Simulation& sim);
      Service* getService(std::string path);
      Publisher* getPublisher(std::string path);
      TriggerType* getTriggerDescriptor(std::string path);
      ~QueryPathParser();
   private:
      Simulation& _sim;
      Queriable* _publisherQueriable;
      std::string _request;
      void parsePath(std::string);
      char* skip(char*);
};
#endif
