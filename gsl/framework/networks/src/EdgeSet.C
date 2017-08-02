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

#include "EdgeSet.h"
#include "ConnectionSet.h"
#include "Edge.h"
#include "Node.h"

EdgeSet::EdgeSet()
{
}


EdgeSet::EdgeSet(EdgeSet* es)
{
   // Shallow copy, Edgeset doesn't own these...
   std::vector<Edge*>::iterator edgeIter = es->_edges.begin();
   std::vector<Edge*>::iterator endEdge = es->_edges.end();
   std::list<GridLayerDescriptor*>::iterator layerIter = es->_layers.begin();
   std::list<GridLayerDescriptor*>::iterator endLayer = es->_layers.end();
   for (; edgeIter != endEdge; edgeIter++) {
      _edges.push_back(*edgeIter);
   }
   for (; layerIter != endLayer; layerIter++) {
      _layers.push_back(*layerIter);
   }
   // _layers = es->_layers;
   // _edges = es->_edges;
}


void EdgeSet::addEdge(Edge* e)
{
   _edges.push_back(e);
   _layers.push_back(e->getPreNode()->getGridLayerDescriptor());
   _layers.push_back(e->getPostNode()->getGridLayerDescriptor());
   _layers.sort();
   _layers.unique();
}


void EdgeSet::reset()
{
   _edges.clear();
   _layers.clear();
}


void EdgeSet::eraseEdge(Edge* e)
{
   std::string modelName = e->getModelName();
   GridLayerDescriptor* pre = e->getPreNode()->getGridLayerDescriptor();
   GridLayerDescriptor* post = e->getPostNode()->getGridLayerDescriptor();

   // first remove edge
   std::vector<Edge*>::iterator iter = _edges.begin();
   std::vector<Edge*>::iterator end = _edges.end();
   for (; iter != end; ++iter) {
      if ((*iter) == e) {
         _edges.erase(iter);
         break;
      }
   }
   if (iter != end) {            // only remove other info if edge was actually in set
      // remove layers entry
      std::list<GridLayerDescriptor*>::iterator gldIter;
      std::list<GridLayerDescriptor*>::iterator gldEnd = _layers.end();
      for (iter = _edges.begin(); iter != end; ++iter) {
         if ((*iter)->getPreNode()->getGridLayerDescriptor() == pre) break;
      }
      if (iter == end) {
         for (gldIter = _layers.begin(); gldIter != gldEnd; ++gldIter) {
            if ((*gldIter) == pre) {
               _layers.erase(gldIter);
               break;
            }
         }
      }
      for (iter = _edges.begin(); iter != end; ++iter) {
         if ((*iter)->getPostNode()->getGridLayerDescriptor() == post) break;
      }
      if (iter == end) {
         for (gldIter = _layers.begin(); gldIter != gldEnd; ++gldIter) {
            if ((*gldIter) == post) {
               _layers.erase(gldIter);
               break;
            }
         }
      }
   }
}


void EdgeSet::addEdges(std::vector<Edge*>* edges)
{
   std::vector<Edge*>::iterator iter = edges->begin();
   std::vector<Edge*>::iterator end = edges->end();
   for (; iter != end; ++iter) {
      Edge* e = (*iter);
      _edges.push_back(e);
      _layers.push_back(e->getPreNode()->getGridLayerDescriptor());
      _layers.push_back(e->getPostNode()->getGridLayerDescriptor());
   }
   _layers.sort();
   _layers.unique();
}


void EdgeSet::addEdges(EdgeSet* edgeSet)
{
   std::vector<Edge*> const & edges = edgeSet->getEdges();
   std::vector<Edge*>::const_iterator iter = edges.begin();
   std::vector<Edge*>::const_iterator end = edges.end();
   for (; iter != end; ++iter) {
      Edge* e = (*iter);
      _edges.push_back(e);
   }
   std::list<GridLayerDescriptor*> layers = edgeSet->getLayers();
   _layers.merge(layers);
   _layers.sort();
   _layers.unique();
}


void EdgeSet::addEdges(ConnectionSet* cs)
{
   ConnectionSet::iterator iter = cs->begin();
   ConnectionSet::iterator end = cs->end();
   for (; iter != end; ++iter) {
      Edge* e = (*iter);
      _edges.push_back(e);
   }
   _layers.push_back(cs->getPrePtr());
   _layers.push_back(cs->getPostPtr());
   _layers.sort();
   _layers.unique();
}


std::vector<Edge*> EdgeSet::getEdgeTypeSet(std::string modelName)
{
   std::vector<Edge*> rvec;
   std::vector<Edge*>::iterator iter = _edges.begin();
   std::vector<Edge*>::iterator end = _edges.end();
   for (; iter != end; ++iter) {
      if ( (*iter)->getModelName() == modelName)
         rvec.push_back(*iter);
   }
   return rvec;
}


std::map<GridLayerDescriptor*, EdgeSet::Position> EdgeSet::getPositions()
{
   std::map<GridLayerDescriptor*, EdgeSet::Position> positions;
   std::map<GridLayerDescriptor*, EdgeSet::Position>::iterator mapIter, mapEnd;
   std::vector<Edge*>::iterator iter = _edges.begin();
   std::vector<Edge*>::iterator end = _edges.end();
   for (; iter != end; ++iter) {
      Edge* e = (*iter);
      GridLayerDescriptor* preGld = e->getPreNode()->getGridLayerDescriptor();
      GridLayerDescriptor* postGld = e->getPostNode()->getGridLayerDescriptor();
      mapEnd = positions.end();
      mapIter = positions.find(preGld);
      if (mapIter != mapEnd) {
         if ((*mapIter).second == _POST) (*mapIter).second = _BOTH;
      }
      else positions[preGld] = _PRE;

      mapIter = positions.find(postGld);
      if (mapIter != mapEnd) {
         if ((*mapIter).second == _PRE) (*mapIter).second = _BOTH;
      }
      else positions[postGld] = _POST;
   }
   return positions;
}


EdgeSet::~EdgeSet()
{
}
