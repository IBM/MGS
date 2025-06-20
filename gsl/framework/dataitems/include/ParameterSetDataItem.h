// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef PARAMETERSETDATAITEM_H
#define PARAMETERSETDATAITEM_H
#include "Copyright.h"

#include "DataItem.h"

#include <memory>
class ParameterSetFactory;
class ParameterSet;

class ParameterSetDataItem : public DataItem
{
   private:
      ParameterSet *_data;

   public:
      static const char* _type;

      ParameterSetDataItem& operator=(const ParameterSetDataItem& DI);

      // Constructors
      ParameterSetDataItem();
      ParameterSetDataItem(std::unique_ptr<ParameterSet>& data);
      ~ParameterSetDataItem();
      ParameterSetDataItem(const ParameterSetDataItem& DI);

      // Utility methods
      void duplicate(std::unique_ptr<DataItem> & r_aptr) const;
      const char* getType() const;

      ParameterSet* getParameterSet() const;
      void setParameterSet(std::unique_ptr<ParameterSet> & ps);
};
#endif
