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

// DataItem is a very generic class that is used for storing
// data. The derived versions of the DataItem class each store a
// particular type of data (int, double, etc), and will allow
// access to the data through the appropriate get/set methods.
// If a get/set method is invoked on an inappropriate type,
// such as calling getInt() on a derived version of DataItem that
// does not store int's, a DataItemException will be thrown.

#ifndef DATAITEM_H_
#define DATAITEM_H_
#include "Copyright.h"

#include <string>
#include <memory>
#include <cstdlib>


#ifdef LINUX
#include <values.h>
#endif

#include <iostream>

class DataItem
{
   public:
      enum Error{CONVERSION_OUT_OF_RANGE, LOSS_OF_PRECISION, COORDS_OUT_OF_RANGE};
      DataItem();
      virtual void duplicate(std::unique_ptr<DataItem> & r_aptr) const =0;
      virtual const char* getType() const =0;
      virtual std::string getString(Error* error=0) const;
      virtual ~DataItem();
};

extern std::ostream& operator<<(std::ostream& os, DataItem& di);
#endif
