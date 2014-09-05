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

#ifndef NDPAIRDATAITEM_H
#define NDPAIRDATAITEM_H
#include "Copyright.h"

#include "DataItem.h"
#include <memory>

class NDPair;

class NDPairDataItem : public DataItem
{
   public:
      static char const* _type;

      NDPairDataItem & operator=(const NDPairDataItem &DI);

      // Constructors
      NDPairDataItem();
      NDPairDataItem(std::auto_ptr<NDPair> data);
      NDPairDataItem(const NDPairDataItem& DI);

      // Destructor
      ~NDPairDataItem();

      // Utility methods
      void duplicate(std::auto_ptr<DataItem> & r_aptr) const;
      const char* getType() const;

      NDPair* getNDPair() const;
      void releaseNDPair(std::auto_ptr<NDPair>& ndp);
      void setNDPair(std::auto_ptr<NDPair>& ndp);
   private:
      NDPair *_data;
      inline void copyContents(const NDPairDataItem& DI);
      inline void destructContents();
};
#endif
