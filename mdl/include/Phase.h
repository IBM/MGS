// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef Phase_H
#define Phase_H
#include "Mdl.h"

#include <memory>
#include <string>
#include <vector>
#include "MemberContainer.h"

class Class;
class PhaseType;
class DataType;
class InterfaceImplementorBase;

class Phase {

   public:
      Phase(const std::string& name, std::unique_ptr<PhaseType>&& phaseType,
	    const std::vector<std::string>& pvn);
      Phase(const Phase& rv);
      Phase& operator=(const Phase& rv);
      virtual void duplicate(std::unique_ptr<Phase>&& rv) const = 0;
      virtual ~Phase();
     
      std::string getGenerateString() const;
      void generateVirtualUserMethod(Class& c) const;    
      void generateUserMethod(Class& c) const;    
      void generateInstancePhaseMethod(
	 Class& c, const std::string& instanceType, 
	 const std::string& componentType,
	 const std::string& workUnitName) const;
      std::string getType() const;

      std::string getName() const {
	 return _name;
      }

      std::string getWorkUnitsMethodBody(
	 const std::string& tab, const std::string& workUnits,
	 const std::string& instanceType, 
	 const std::string& componentType) const;

      std::string getInitializePhaseMethodBody() const;

      void setPackedVariables(const InterfaceImplementorBase& base);
      std::vector<std::string>& getPackedVariableNames() {return _packedVariableNames;}
      std::vector<const DataType*>& getPackedVariables() {return _packedVariables;}

      bool hasPackedVariables() const {
	 return (_packedVariableNames.size() > 0);
      }

      std::string getAddVariableNamesForPhase(const std::string& tab) const;

   protected:
      std::string _name;
      PhaseType* _phaseType;
      std::vector<std::string> _packedVariableNames;
      // not owned
      std::vector<const DataType*> _packedVariables; 
      void generateInternalUserMethod(Class& c) const;    
      virtual std::string getInternalType() const = 0;

   private:
      void copyOwnedHeap(const Phase& rv);
      void destructOwnedHeap();
};


#endif // Phase_H
