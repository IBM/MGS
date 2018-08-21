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

#ifndef ComputeTime_H
#define ComputeTime_H
#include "Mdl.h"

#include <memory>
#include <string>
#include <vector>
#include "MemberContainer.h"

class Class;
class ComputeTimeType;
class DataType;
class InterfaceImplementorBase;

class ComputeTime {

   public:
      ComputeTime(const std::string& name, std::auto_ptr<ComputeTimeType>& computeTimeType,
	    const std::vector<std::string>& pvn);
      ComputeTime(const ComputeTime& rv);
      ComputeTime& operator=(const ComputeTime& rv);
      virtual void duplicate(std::auto_ptr<ComputeTime>& rv) const = 0;
      virtual ~ComputeTime();
     
      std::string getGenerateString() const;
      void generateVirtualUserMethod(Class& c) const;    
      void generateUserMethod(Class& c) const;    
      void generateInstanceComputeTimeMethod(
	 Class& c, const std::string& instanceType, 
	 const std::string& componentType) const;
      std::string getType() const;

      std::string getName() const {
	 return _name;
      }

      std::string getWorkUnitsMethodBody(
	 const std::string& tab, const std::string& workUnits,
	 const std::string& instanceType, 
	 const std::string& componentType) const;

      std::string getInitializeComputeTimeMethodBody() const;

      void setPackedVariables(const InterfaceImplementorBase& base);
      std::vector<std::string>& getPackedVariableNames() {return _packedVariableNames;}
      std::vector<const DataType*>& getPackedVariables() {return _packedVariables;}

      bool hasPackedVariables() const {
	 return (_packedVariableNames.size() > 0);
      }

      std::string getAddVariableNamesForComputeTime(const std::string& tab) const;

   protected:
      std::string _name;
      ComputeTimeType* _computeTimeType;
      std::vector<std::string> _packedVariableNames;
      // not owned
      std::vector<const DataType*> _packedVariables; 
      void generateInternalUserMethod(Class& c, bool pureVirtual) const;    
      virtual std::string getInternalType() const = 0;

   private:
      void copyOwnedHeap(const ComputeTime& rv);
      void destructOwnedHeap();
};

#endif // ComputeTime_H
