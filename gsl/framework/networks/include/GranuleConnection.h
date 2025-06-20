// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
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
