// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef ArgumentListHelper_H
#define ArgumentListHelper_H
#include "Copyright.h"

#include "DataItem.h"
#include "GslContext.h"
#include "C_argument_list.h"
#include "C_type_specifier.h"
#include <memory>

class ArgumentListHelper
{
   public:
      void getDataItem(std::unique_ptr<DataItem>& dataItem
		  , GslContext* c
		  , C_argument_list* argumentList
		  , C_type_specifier* typeSpec);
};
#endif
