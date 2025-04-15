// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef INDEXSET_H
#define INDEXSET_H
#include "Copyright.h"

#include <vector>


class Grid;

class IndexSet
{
   public:
      // constructors/destructor
      IndexSet(){}
      IndexSet(std::vector<int>& begin, std::vector<int>& end);
      IndexSet(const IndexSet&);

      ~IndexSet();

      IndexSet& operator=(const IndexSet&);

      std::vector<int>& getBeginCoords();
      std::vector<int>& getEndCoords();

   private:
      std::vector<int> _begin;
      std::vector<int> _end;

};
#endif
