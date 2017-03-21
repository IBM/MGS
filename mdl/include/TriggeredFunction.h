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

#ifndef TriggeredFunction_H
#define TriggeredFunction_H
#include "Mdl.h"

#include <memory>
#include <string>

class Class;

class TriggeredFunction {

   public:
      enum RunType {_SERIAL, _PARALLEL};

      TriggeredFunction(const std::string& name, RunType runType);
      virtual void duplicate(std::auto_ptr<TriggeredFunction>& rv) const = 0;
      virtual ~TriggeredFunction();

      std::string getNameToCallerCodeString(
	 const std::string& triggerableCallerName, 
	 const std::string& className) const;

      const std::string& getName() const {
	 return _name;
      }
      
      std::string getString() const;

      void addEventMethodToClass(Class& instance, bool pureVirtual) const;

   protected:
      std::string _name;
      RunType _runType;

      std::string getType() const;
      std::string getReturnString() const;
      virtual std::string getTab() const = 0;

      virtual std::string getTriggerableType (
	 const std::string& modelName) const = 0;

};

#endif // TriggeredFunction_H
