// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
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
      virtual void duplicate(std::unique_ptr<GranuleMapper>&& dup) const {assert(0);}
      virtual std::string getName() = 0;

   protected:
      std::vector<Granule> _granules;
      unsigned _index;
};

#endif
