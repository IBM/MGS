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
