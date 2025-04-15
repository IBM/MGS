// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef ToolBase_H
#define ToolBase_H
#include "Mdl.h"

#include "Generatable.h"
#include "MemberContainer.h"

#include <string>
#include <memory>

class DataType;

class ToolBase : public Generatable {
   public:
      ToolBase(const std::string& fileName);
      virtual void duplicate(std::unique_ptr<Generatable>&& rv) const =0;
      virtual std::string getType() const =0;
      virtual void generate() const;
      virtual std::string generateExtra() const;
      virtual std::string generateTitleExtra() const;
      virtual ~ToolBase();        
      const std::string& getName() const;
      void setName(const std::string& name);
      

      MemberContainer<DataType> _initializeArguments;
      bool _userInitialization;

   protected:
      virtual std::string getModuleName() const;
      void generateInitializer(const std::string& type, 
			       MemberContainer<DataType>& members,
			       bool userInit);
      void generateInitArgs();

   private:
      std::string _name;
};

#endif // ToolBase_H
