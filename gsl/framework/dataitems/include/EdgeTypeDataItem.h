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

#ifndef EDGETYPEDATAITEM_H
#define EDGETYPEDATAITEM_H
#include "Copyright.h"

#include "DataItem.h"

class EdgeType;

class EdgeTypeDataItem : public DataItem
{
   protected:
      DataItem & assign(const DataItem &);

   private:
      EdgeType *_data;

   public:
      static char const* _type;

      EdgeTypeDataItem(EdgeType *data = 0);
      EdgeTypeDataItem (const EdgeTypeDataItem& DI);

      virtual EdgeTypeDataItem& operator=(const EdgeTypeDataItem& DI);

      void duplicate(std::auto_ptr<DataItem> & r_aptr) const;
      const char* getType() const;

      void setEdgeType(EdgeType*);
      EdgeType* getEdgeType() const;

};
#endif
