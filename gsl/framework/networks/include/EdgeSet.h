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

#ifndef EDGESET_H
#define EDGESET_H
#include "Copyright.h"

#include <vector>
#include <list>
#include <string>
#include <map>


class Edge;
class GridLayerDescriptor;
class ConnectionSet;

class EdgeSet
{
   public:
      enum Position {_PRE, _POST, _BOTH};
      EdgeSet();
      EdgeSet(EdgeSet*);
      std::vector<Edge*>& getEdges() {return _edges;}
      std::map<GridLayerDescriptor*, EdgeSet::Position> getPositions();
      void addEdge(Edge* e);
      void eraseEdge(Edge* e);
      void addEdges(std::vector<Edge*>* v);
      void addEdges(EdgeSet* cs);
      void addEdges(ConnectionSet* cs);
      void reset();
      const std::list<GridLayerDescriptor*>& getLayers() const {return _layers;}
      std::vector<Edge*> getEdgeTypeSet(std::string modelName);
      ~EdgeSet();
   private:
      std::vector<Edge*> _edges;
      std::list<GridLayerDescriptor*> _layers;
};
#endif
