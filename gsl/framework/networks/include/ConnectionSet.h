// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef CONNECTIONSET_H
#define CONNECTIONSET_H
#include "Copyright.h"

#include <vector>
#include <string>

class Edge;
class GridLayerDescriptor;
class Publisher;

class ConnectionSet : public std::vector<Edge*>
{
   friend class Repertoire;      
   // Repertoire is a friend so it can have 
   // access to the blessed function addEdge which does not do any error 
   // checking for whether or not edge is valid.

   public:
      ConnectionSet(GridLayerDescriptor* pre, GridLayerDescriptor* post);
      ConnectionSet(ConnectionSet* cs);
      GridLayerDescriptor* getPrePtr();
      GridLayerDescriptor* getPostPtr();
      void addConnection(Edge* e);
      std::string getName();
      virtual ~ConnectionSet();

   private:
      void addEdge(Edge* e);
      GridLayerDescriptor *_pre, *_post;
      std::string _name;
};

#endif
