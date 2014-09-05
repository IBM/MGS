// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BMC-YKT-08-23-2011-2
//
// (C) Copyright IBM Corp. 2005-2014  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

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
      virtual void duplicate(std::auto_ptr<Generatable>& rv) const;
      virtual void duplicate(std::auto_ptr<Interface>& rv) const;
      virtual void generate() const;
      virtual ~Interface();        
      const std::string& getName() const;
      void setName(const std::string& name);
      void addProducerMethods(Class& c);

      void addDataTypeToMembers(std::auto_ptr<DataType>& dataType);
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
