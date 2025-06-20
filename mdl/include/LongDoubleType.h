// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef LongDoubleType_H
#define LongDoubleType_H
#include "Mdl.h"

#include <string>
#include <memory>
#include "DataType.h"

class LongDoubleType : public DataType {
   public:
      virtual void duplicate(std::unique_ptr<DataType>&& rv) const;
      virtual bool isBasic() const;
      virtual ~LongDoubleType();        

      virtual std::string getDescriptor() const;
      virtual std::string getArrayDataItemString() const;
      virtual std::string getCapitalDescriptor() const;
      virtual std::string getDataItemString() const;
      virtual std::string getDataItemFunctionString() const;
      virtual std::string getArrayInitializerString(
	 const std::string& name,
	 const std::string& arrayName,
	 int level) const;
      virtual std::string getInitializationDataItemString() const;
};

#endif // LongDoubleType_H
