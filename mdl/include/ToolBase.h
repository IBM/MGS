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
      virtual void duplicate(std::auto_ptr<Generatable>& rv) const =0;
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
