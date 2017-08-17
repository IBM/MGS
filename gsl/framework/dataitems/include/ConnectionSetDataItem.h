// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BCM-YKT-11-19-2015
//
// (C) Copyright IBM Corp. 2005-2015  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#ifndef CONNECTIONSETDATAITEM_H
#define CONNECTIONSETDATAITEM_H
#include "Copyright.h"

#include "DataItem.h"

class ConnectionSet;

class ConnectionSetDataItem : public DataItem
{
   private:
      ConnectionSet *_data;

   public:
      static char const * _type;

      virtual ConnectionSetDataItem& operator=(const ConnectionSetDataItem& DI);

      ConnectionSetDataItem(ConnectionSet *data = 0);
      ConnectionSetDataItem(const ConnectionSetDataItem& DI);

      void duplicate(std::auto_ptr<DataItem> & r_aptr) const;
      const char* getType() const;

      ConnectionSet* getConnectionSet(Error* error=0) const;
      void setConnectionSet(ConnectionSet* ns, Error* error=0);
};
#endif
