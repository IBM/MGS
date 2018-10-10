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
