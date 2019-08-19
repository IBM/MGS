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

#ifndef GRANULEMAPPERBASE_H
#define GRANULEMAPPERBASE_H
#include "Copyright.h"

#include "GranuleMapper.h"
#include "Granule.h"

#include <list>
#include <string>
#include <vector>
#include <deque>
#include <cassert>

class Graph;

class GranuleMapperBase : public GranuleMapper
{
   public:
      GranuleMapperBase();
      virtual unsigned getNumberOfGranules() {
         return _granules.size();
      }
      virtual void setGraphId(unsigned& current);
      virtual void initializeGraph(Graph* graph);
      virtual void setGlobalGranuleIds(unsigned& id);
      virtual ~GranuleMapperBase();
      virtual unsigned getIndex() {return _index;}
      virtual void setIndex(unsigned index) {_index=index;}
      virtual void duplicate(std::unique_ptr<GranuleMapper>& dup) const {assert(0);}
      virtual std::string getName() = 0;

   protected:
      std::vector<Granule> _granules;
      unsigned _index;
};

#endif
