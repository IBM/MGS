// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BCM-YKT-11-19-2015
//
// (C) Copyright IBM Corp. 2005-2015  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

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
