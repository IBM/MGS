// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef EDGETYPE_H
#define EDGETYPE_H
#include "Copyright.h"

#define EDGE_ALLOCATION 1000

#include <memory>
#include <string>
#include <map>


class ConnectionIncrement;
class ParameterSet;
class Edge;

class EdgeType
{

   public:
      virtual void getInitializationParameterSet(std::unique_ptr<ParameterSet> & r_aptr) =0;
      virtual Edge* getEdge() =0;
      virtual std::string getModelName() =0;
      virtual ~EdgeType() {}
      
//      virtual std::map<std::string, ConnectionIncrement>* getComputeCost() const = 0; // modified by Jizhu Lu on 01/30/2006
//      virtual ConnectionIncrement* getComputeCost() const = 0; // modified by Jizhu Lu on 01/30/2006
      virtual ConnectionIncrement* getComputeCost() = 0; // modified by Jizhu Lu on 01/30/2006

      // Will be = 0 when CG is complete
      virtual void getInAttrParameterSet(
	 std::unique_ptr<ParameterSet> & r_aptr) {};
      virtual void getOutAttrParameterSet(
	 std::unique_ptr<ParameterSet> & r_aptr) {};

};
#endif
