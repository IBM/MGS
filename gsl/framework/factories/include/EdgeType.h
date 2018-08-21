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
      virtual void getInitializationParameterSet(std::auto_ptr<ParameterSet> & r_aptr) =0;
      virtual Edge* getEdge() =0;
      virtual std::string getModelName() =0;
      virtual ~EdgeType() {}
      
//      virtual std::map<std::string, ConnectionIncrement>* getComputeCost() const = 0; // modified by Jizhu Lu on 01/30/2006
//      virtual ConnectionIncrement* getComputeCost() const = 0; // modified by Jizhu Lu on 01/30/2006
      virtual ConnectionIncrement* getComputeCost() = 0; // modified by Jizhu Lu on 01/30/2006

      // Will be = 0 when CG is complete
      virtual void getInAttrParameterSet(
	 std::auto_ptr<ParameterSet> & r_aptr) {};
      virtual void getOutAttrParameterSet(
	 std::auto_ptr<ParameterSet> & r_aptr) {};

};
#endif
