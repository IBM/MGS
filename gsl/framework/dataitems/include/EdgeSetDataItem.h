// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef EDGESETDATAITEM_H
#define EDGESETDATAITEM_H
#include "Copyright.h"

#include "TriggerableDataItem.h"
#include <vector>

class EdgeSet;
class ConnectionSet;
class Triggerable;

class EdgeSetDataItem : public TriggerableDataItem
{
   private:
      EdgeSet *_data;

   public:
      static char const * _type;

      virtual EdgeSetDataItem& operator=(const EdgeSetDataItem& DI);

      EdgeSetDataItem();
      EdgeSetDataItem(std::unique_ptr<EdgeSet> data);
      EdgeSetDataItem(const EdgeSetDataItem& DI);
      ~EdgeSetDataItem();

      void duplicate(std::unique_ptr<DataItem> & r_aptr) const;
      const char* getType() const;

      EdgeSet* getEdgeSet(Error* error=0) const;
      void setEdgeSet(EdgeSet* ns, Error* error=0);
      void setEdgeSet(ConnectionSet* cs, Error* error=0);

      virtual std::vector<Triggerable*> getTriggerables();
};
#endif
