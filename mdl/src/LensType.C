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

#include "LensType.h"
#include "DataType.h"
#include "SyntaxErrorException.h"
#include "Constants.h"
#include <string>
#include <sstream>
#include <memory>
#include <vector>

void LensType::duplicate(std::unique_ptr<DataType>&& rv) const
{
   rv.reset(new LensType(*this));
}

void LensType::setPointer(bool pointer) 
{
   if (pointer == false) {
      throw SyntaxErrorException(
	 getDescriptor() 
	 + "Type is a LensType and must be defined as a pointer.");
   }
   _pointer = pointer;
}

std::string LensType::getDescriptor() const
{
   return "LensTypeBaseClass";
}

std::string LensType::getHeaderString(
   std::vector<std::string>& arrayTypeVec) const
{
   return "\"" + getDescriptor() + ".h\"";
}

LensType::~LensType() 
{
}

bool LensType::isLegitimateDataItem() const
{
   return isPointer();
}

std::string LensType::getServiceString(const std::string& tab) const
{
   // Lens types don't have services.
   return "";
}

std::string LensType::getOptionalServiceString(const std::string& tab) const
{
   // Lens types don't have services.
   return "";
}

std::string LensType::getServiceNameString(const std::string& tab) const
{
   // Lens types don't have services.
   return "";
}

std::string LensType::getServiceDescriptionString(const std::string& tab) const
{
   // Lens types don't have services.
   return "";
}

std::string LensType::getServiceDescriptorString(const std::string& tab) const
{
   // Lens types don't have services.
   return "";
}
