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
      EdgeSetDataItem(std::auto_ptr<EdgeSet> data);
      EdgeSetDataItem(const EdgeSetDataItem& DI);
      ~EdgeSetDataItem();

      void duplicate(std::auto_ptr<DataItem> & r_aptr) const;
      const char* getType() const;

      EdgeSet* getEdgeSet(Error* error=0) const;
      void setEdgeSet(EdgeSet* ns, Error* error=0);
      void setEdgeSet(ConnectionSet* cs, Error* error=0);

      virtual std::vector<Triggerable*> getTriggerables();
};
#endif
