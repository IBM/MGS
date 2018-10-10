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

#ifndef REPERTOIREFACTORYDATAITEM_H
#define REPERTOIREFACTORYDATAITEM_H
#include "Copyright.h"

#include "DataItem.h"
#include <memory>

class RepertoireFactory;

class RepertoireFactoryDataItem : public DataItem
{
   private:
      RepertoireFactory *_data;

   public:
      static const char* _type;

      RepertoireFactoryDataItem & operator=(const RepertoireFactoryDataItem &);
      // Constructors
      RepertoireFactoryDataItem();
      RepertoireFactoryDataItem(std::unique_ptr<RepertoireFactory> data);
      RepertoireFactoryDataItem(const RepertoireFactoryDataItem& DI);
      ~RepertoireFactoryDataItem();

      // Utility methods
      void duplicate(std::unique_ptr<DataItem> & r_aptr) const;
      const char* getType() const;

      RepertoireFactory* getFactory(Error* error=0) const;
      void setFactory(std::unique_ptr<RepertoireFactory>&, Error* error=0);
};
#endif
