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
