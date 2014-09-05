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

#ifndef GraphConnection_H
#define GraphConnection_H
#include "Copyright.h"

#include <string>

class GraphConnection
{
   public:
      GraphConnection(unsigned graphId, float weight);
      GraphConnection();

      void addWeight(float weight) {
	 _weight += weight;
      }
      void setWeight(float weight) {
	 _weight=weight;
      }
      float getWeight() const {
	 return _weight;
      }
      bool operator<(const GraphConnection& rv) const {
	 return _graphId < rv._graphId; 
      }
      void setGraphId(unsigned graphId) {
	 _graphId=graphId;
      }
      unsigned getGraphId() const {
	 return _graphId;
      }

   private:
      unsigned _graphId;
      float _weight;
};

extern std::ostream& operator<<(std::ostream& os, const GraphConnection& inp);
extern std::istream& operator>>(std::istream& is, GraphConnection& inp);

#endif
