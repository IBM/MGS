// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef Interface_H
#define Interface_H
#include "Mdl.h"

#include <string>
#include <memory>
#include "Generatable.h"
#include "MemberContainer.h"
#include "Class.h"

class DataType;

class Interface : public Generatable {
   public:
      Interface(const std::string& fileName);
      virtual void duplicate(std::unique_ptr<Generatable>&& rv) const;
      virtual void duplicate(std::unique_ptr<Interface>&& rv) const;
      virtual void generate() const;
      virtual ~Interface();        
      const std::string& getName() const;
      void setName(const std::string& name);
      void addProducerMethods(Class& c);

      void addDataTypeToMembers(std::unique_ptr<DataType>&& dataType);
      const MemberContainer<DataType>& getMembers() {
	 return _members;
      } 

   protected:
      virtual std::string getModuleName() const;
      virtual std::string getModuleTypeName() const;
      virtual void internalGenerateFiles();
      void generateInstance();

   private:
      MemberContainer<DataType> _members;
      std::string _name;
};

#endif // Interface_H
