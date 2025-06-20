// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
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
