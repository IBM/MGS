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

#include "Granule.h"
#include "GranuleConnection.h"
#include "ConnectionIncrement.h"
#include "Graph.h"
#include <cassert>
#include <sstream>
#include <iostream>
#ifdef HAVE_MPI
#include <mpi.h>
#endif
#include <cassert>

Granule::Granule()
   : _globalGranuleId(0), _graphId(0), _partitionId(0), 
     _depends(0)   
{   
    _computeCost = new ConnectionIncrement();
}

unsigned Granule::getPartitionId() const 
{
   int retval = _partitionId;
   if (_depends) {
      retval = _depends->getPartitionId();
   }
   return retval;
}

void Granule::setDepends(Granule* depends) 
{
   if (_depends == 0) {
      _depends = depends;      
   } else {
      assert(depends == _depends);
   }
}

void Granule::addConnection(Granule* post, float weight)
{
   /* when add a granule connection, we should determine that whether the
      two nodes are in the same granule. If yes, we need to add the computation time,
      the memory bytes but should not add communication bytes. If not, then
      it depends on whether the two granules will be assigned to the same graph Id.
      Granules with the same graph ID will be assigned to the same memory space.
      in that case, the computation time, the memory bytes should be added, but
      communication bytes should not. If two granules are with different graph
      IDs, then only the communication bytes need to be added. However, if we
      do not add them how do we know which two granules should have the same
      graph ID?   */
   GranuleConnection connection(post, weight);

   std::set<GranuleConnection>::iterator it = _connections.find(connection); 
   if (it == _connections.end()) {
      _connections.insert(connection);
   } else {
      // Weight isn't a key value that will change the nature of the
      // binary search tree, so I can do a const cast... --sgc
      const_cast<GranuleConnection&>(*it).addWeight(weight);
   }
}

void Granule::addGraphConnection(unsigned graphId, float weight)
{
   // Due to separability constraints these two may have fallen into the
   // same granule, if so ignore
   if (_graphId == graphId) return;

   GraphConnection connection(graphId, weight);
   
   std::set<GraphConnection>::iterator it = _graphConnections.find(connection);
   if (it == _graphConnections.end()) {
      _graphConnections.insert(connection);
   } else {
      // Weight isn't a key value that will change the nature of the
      // binary search tree, so I can do a const cast... --sgc
      const_cast<GraphConnection&>(*it).addWeight(weight);
   }
}

void Granule::setGraphId(unsigned& current)
{

   if (_depends == NULL) {
      _graphId = current++;      
   } else {
      assert(_depends != NULL);
      _graphId = _depends->getGraphId();
   }
}

void Granule::initializeGraph(Graph* graph)
{
   Granule* which = this;

   int density = 1;                            // added by Jizhu Lu on 01/30/2006

   if (_depends) {
      _depends->addComputeCost(density, _computeCost);
      which = _depends;
   } else { // only set this if you don't have a dependency
      graph->setNode(_graphId, this);
   }
   
   std::set<GranuleConnection>::iterator it, end = _connections.end(); 
   for (it = _connections.begin(); it != end; ++it) {
      which->addGraphConnection(it->getGranule()->getGraphId(), it->getWeight());
   }   
}

std::ostream& operator<<(std::ostream& os, const Granule& inp)
{
   os << inp.getGraphId() << " " << inp.getPartitionId() << " ";

   std::set<GraphConnection>::const_iterator it, end = inp.getGraphConnections().end();

   os << inp.getGraphConnections().size();

   for (it = inp.getGraphConnections().begin(); it != end; ++it) {      
     os << (*it);
   }
   return os;
}

std::istream& operator>>(std::istream& is, Granule& inp)
{
  unsigned gid;
  is >> gid;
  inp.setGraphId(gid);

  unsigned pid;
  is >> pid;
  inp.setPartitionId(pid);

  std::set<GraphConnection>& connections =  inp.getModifiableGraphConnections();
  connections.clear();

  unsigned count;
  is >> count;

  for (unsigned i=0; i<count; ++i) {
    GraphConnection gc;
    is >> gc;
    connections.insert(gc);
  }
  return is;
}

Granule::~Granule()
{
}
