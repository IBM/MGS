// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef Functor_H
#define Functor_H
#include "Mdl.h"

#include <string>
#include <memory>
#include "ToolBase.h"
#include "Generatable.h"
#include "MemberContainer.h"
#include "Method.h"
#include "DataType.h"

class Functor : public ToolBase {
   public:
      Functor(const std::string& fileName);
      Functor(const Functor& rv);
      Functor& operator=(const Functor& rv);
      virtual void duplicate(std::unique_ptr<Generatable>&& rv) const;
      virtual std::string getType() const;
      virtual std::string generateExtra() const;
      virtual std::string generateTitleExtra() const;
      void setReturnType(std::unique_ptr<DataType>&& ret);
      const std::string& getCategory() const;
      void setCategory(const std::string& category);
      virtual ~Functor();        

      virtual std::string getTypeDescription();

      MemberContainer<DataType> *_executeArguments;
      bool _userExecute;

   protected:
      void copyOwnedHeap(const Functor& rv);
      void destructOwnedHeap();
      virtual std::string getModuleTypeName() const;
      virtual void internalGenerateFiles();
      void generateExecArgs();
      void generateInstanceBase();
      void generateInstance();

   private:
      void createInitMethod(std::unique_ptr<Method>&& method, 
			    const MemberContainer<DataType>& args,
			    const std::string& funcName, 
			    const std::string& attName,
			    bool userInit, bool hasRetVal);
      void createUserMethod(std::unique_ptr<Method>&& method, 
			    const MemberContainer<DataType>& args,
			    const std::string& funcName, 
			    const std::string& retType,
			    bool userInit, bool pureVirtual);
      DataType* _returnType;
      std::string _category;
};

#endif // Functor_H
