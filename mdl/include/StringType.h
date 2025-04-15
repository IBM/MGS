// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef StringType_H
#define StringType_H
#include "Mdl.h"

#include <string>
#include <memory>
#include <vector>
#include "DataType.h"

class StringType : public DataType {
   public:
      virtual void duplicate(std::unique_ptr<DataType>&& rv) const;
      virtual ~StringType();        
      
      virtual std::string getDescriptor() const;
      virtual std::string getHeaderString(
	 std::vector<std::string>& arrayTypeVec) const;
   protected:
      virtual std::string getDataItemFunctionString() const;    
};

#endif // StringType_H
