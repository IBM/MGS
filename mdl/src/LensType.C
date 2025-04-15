// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
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
