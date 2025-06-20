// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef SERVICE_H
#define SERVICE_H
#include "Copyright.h"

#include <vector>
#include <string>
#include <memory>

class Key;
class DataItem;

class Service
{

   public:
      virtual std::string getName() const =0;
                                 // human readable
      virtual std::string getDescription() const =0;
      virtual std::string getDataItemDescription() const =0;
      virtual std::string getStringValue() const = 0;      
      virtual void setStringValue(const std::string& value) = 0;      

      virtual void duplicate(std::unique_ptr<Service>& dup) const = 0;
      virtual ~Service() {}


};
#endif
