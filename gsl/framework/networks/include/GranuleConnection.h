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

#ifndef GranuleConnection_H
#define GranuleConnection_H
#include "Copyright.h"

class Granule;

class GranuleConnection
{
   public:
      GranuleConnection(Granule* granule, float weight);

      void addWeight(float weight) {
	 _weight += weight;
      }
      float getWeight() const {
	 return _weight;
      }
      bool operator<(const GranuleConnection& rv) const {
	 return _granule < rv._granule; 
      }
      const Granule* getGranule() const {
	 return _granule;
      }
      
   private:
      Granule* _granule;
      float _weight;
};

#endif
