// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "GslType.h"
#include "DataType.h"
#include "SyntaxErrorException.h"
#include "Constants.h"
#include <string>
#include <sstream>
#include <memory>
#include <vector>

void GslType::duplicate(std::unique_ptr<DataType>&& rv) const
{
   rv.reset(new GslType(*this));
}

void GslType::setPointer(bool pointer) 
{
   if (pointer == false) {
      throw SyntaxErrorException(
	 getDescriptor() 
	 + "Type is a GslType and must be defined as a pointer.");
   }
   _pointer = pointer;
}

std::string GslType::getDescriptor() const
{
   return "GslTypeBaseClass";
}

std::string GslType::getHeaderString(
   std::vector<std::string>& arrayTypeVec) const
{
   return "\"" + getDescriptor() + ".h\"";
}

GslType::~GslType() 
{
}

bool GslType::isLegitimateDataItem() const
{
   return isPointer();
}

std::string GslType::getServiceString(const std::string& tab) const
{
   // Gsl types don't have services.
   return "";
}

std::string GslType::getOptionalServiceString(const std::string& tab) const
{
   // Gsl types don't have services.
   return "";
}

std::string GslType::getServiceNameString(const std::string& tab) const
{
   // Gsl types don't have services.
   return "";
}

std::string GslType::getServiceDescriptionString(const std::string& tab) const
{
   // Gsl types don't have services.
   return "";
}

std::string GslType::getServiceDescriptorString(const std::string& tab) const
{
   // Gsl types don't have services.
   return "";
}
