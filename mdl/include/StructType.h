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

#ifndef StructType_H
#define StructType_H
#include "Mdl.h"

#include <string>
#include <memory>
#include "DataType.h"
#include "Generatable.h"
#include "MemberContainer.h"


class StructType : public DataType, public Generatable {
   public:
      // Structtype is not always only generatable so the filename 
      // can be == "" for some instantiations.
      StructType(const std::string& fileName = "");
      virtual void duplicate(std::auto_ptr<StructType>& rv) const;
      virtual void duplicate(std::auto_ptr<DataType>& rv) const;
      virtual void duplicate(std::auto_ptr<Generatable>& rv) const;
      virtual void generate() const;
      std::string getInAttrPSetStr() const;
      std::string getOutAttrPSetStr() const;
      virtual ~StructType();        
      const std::string& getTypeName() const;
      void setTypeName(const std::string& type);

      MemberContainer<DataType> _members;

      virtual std::string getDescriptor() const;
      virtual std::string getHeaderString(
	 std::vector<std::string>& arrayTypeVec) const;
      virtual std::string getDataItemString() const;

      virtual bool isTemplateMarshalled() const {return false;}
      virtual bool isTemplateDemarshalled() const {return false;}
      virtual bool isSuitableForInterface() const;
      virtual void getSubStructDescriptors(std::set<std::string>& subStructTypes) const;

   protected:
      virtual bool isSuitableForFlatDemarshaller() const;
      virtual bool isSuitableForFlatMarshaller() const {return isSuitableForFlatDemarshaller();}
      virtual std::string getDataItemFunctionString() const;

      // This method produces the checking code, and updates the name 
      // to the type checked  struct. This method does nothing, 
      // it is overriden in StructType. The overriden method is going 
      // to change name, that is why it is not const.
      virtual std::string checkIfStruct(const std::string& name, 
					const std::string& tab,
					std::string& newName) const;

      // This method returns generated code. The returned value is 
      // used in the right hand side of the initialization of the DataType.
      virtual std::string getDataFromVariable(const std::string& name) const;

      virtual std::string getModuleName() const;
      virtual std::string getModuleTypeName() const;
      virtual void internalGenerateFiles();
      void generateInstance();
      void generateFlatDemarshaller();
      void generateDemarshaller();
      void generateFlatMarshaller();
      void generateMarshaller();

   private:
      std::string getMarshallMethodFunctionBody() const;
      std::string getGetBlocksMethodFunctionBody() const;

      std::string _typeName;
      std::string getPSetStr(const std::string& which) const;

};

#endif // StructType_H
