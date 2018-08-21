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
