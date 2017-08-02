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
      void duplicate(std::auto_ptr<DataItem> & r_aptr) const;
      const char* getType() const;

      Edge* getEdge() const;
      void setEdge(Edge* e);
      std::string getString(Error* error=0) const;

};
#endif
