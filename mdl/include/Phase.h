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
      Phase(const std::string& name, std::auto_ptr<PhaseType>& phaseType,
	    const std::vector<std::string>& pvn);
      Phase(const Phase& rv);
      Phase& operator=(const Phase& rv);
      virtual void duplicate(std::auto_ptr<Phase>& rv) const = 0;
      virtual ~Phase();
     
      std::string getGenerateString() const;
      void generateVirtualUserMethod(Class& c) const;    
      void generateUserMethod(Class& c) const;    
      void generateInstancePhaseMethod(
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
