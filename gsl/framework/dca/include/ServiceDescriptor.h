// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef ServiceDescriptor_H
#define ServiceDescriptor_H
#include "Copyright.h"

#include <vector>
//#include <string>
#include <memory>

class Key;
class DataItem;

class ServiceDescriptor
{

   public:
      ServiceDescriptor(const char* name, const char* description, 
			const char* dataItemDescription)
	 : _name(name), _description(description), 
	   _dataItemDescription(dataItemDescription)
	 {}
      const char* getName() const {
	 return _name;
      }
      const char* getDescription() const {
	 return _description;
      }
      const char* getDataItemDescription() const {
	 return _dataItemDescription;
      }
   private:
      const char* _name;
      const char* _description;
      const char* _dataItemDescription;

};
#endif
