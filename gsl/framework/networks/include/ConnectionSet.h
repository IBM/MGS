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
