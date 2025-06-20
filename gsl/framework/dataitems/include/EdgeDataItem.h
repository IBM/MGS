// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef EDGEDATAITEM_H
#define EDGEDATAITEM_H
#include "Copyright.h"

#include "DataItem.h"

class Edge;

class EdgeDataItem : public DataItem
{
   private:
      Edge *_edge;

   public:
      static const char* _type;

      virtual EdgeDataItem& operator=(const EdgeDataItem& DI);

      // Constructors
      EdgeDataItem(Edge *edge = 0);
      EdgeDataItem(const EdgeDataItem& DI);

      // Destructor
      ~EdgeDataItem();

      // Utility methods
      void duplicate(std::unique_ptr<DataItem> & r_aptr) const;
      const char* getType() const;

      Edge* getEdge() const;
      void setEdge(Edge* e);
      std::string getString(Error* error=0) const;

};
#endif
