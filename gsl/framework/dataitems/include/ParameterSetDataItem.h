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
      ParameterSetDataItem(std::auto_ptr<ParameterSet> data);
      ~ParameterSetDataItem();
      ParameterSetDataItem(const ParameterSetDataItem& DI);

      // Utility methods
      void duplicate(std::auto_ptr<DataItem> & r_aptr) const;
      const char* getType() const;

      ParameterSet* getParameterSet() const;
      void setParameterSet(std::auto_ptr<ParameterSet> & ps);
};
#endif
