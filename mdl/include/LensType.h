// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef LensType_H
#define LensType_H
#include "Mdl.h"

#include <string>
#include <memory>
#include <vector>
#include "DataType.h"

class LensType : public DataType {
   using DataType::getServiceNameString;  // Make base class method visible
   using DataType::getServiceDescriptionString;  // Make base class method visible
   public:
      virtual void duplicate(std::unique_ptr<DataType>&& rv) const;
      virtual ~LensType();        
      virtual void setPointer(bool pointer);

      virtual std::string getDescriptor() const;
      virtual std::string getHeaderString(
	 std::vector<std::string>& arrayTypeVec) const;
      virtual bool isLegitimateDataItem() const;

      virtual bool isSuitableForInterface() const {
	 return false;
      }
      virtual std::string getServiceString(const std::string& tab) const;

      virtual std::string getOptionalServiceString(
	 const std::string& tab) const;

      // This function returns code for the name of the service.
      virtual std::string getServiceNameString(const std::string& tab) const;

      // This function returns code for the description of the service.
      virtual std::string getServiceDescriptionString(
	 const std::string& tab) const;

      // This function returns code for setting up the ServiceDescriptor.
      virtual std::string getServiceDescriptorString(
	 const std::string& tab) const;
};

#endif // LensType_H
