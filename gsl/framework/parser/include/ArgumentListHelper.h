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

#ifndef ArgumentListHelper_H
#define ArgumentListHelper_H
#include "Copyright.h"

#include "DataItem.h"
#include "LensContext.h"
#include "C_argument_list.h"
#include "C_type_specifier.h"
#include <memory>

class ArgumentListHelper
{
   public:
      void getDataItem(std::auto_ptr<DataItem>& dataItem
		  , LensContext* c
		  , C_argument_list* argumentList
		  , C_type_specifier* typeSpec);
};
#endif
