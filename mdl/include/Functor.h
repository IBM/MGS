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
      virtual void duplicate(std::auto_ptr<Generatable>& rv) const;
      virtual std::string getType() const;
      virtual std::string generateExtra() const;
      virtual std::string generateTitleExtra() const;
      void setReturnType(std::auto_ptr<DataType>& ret);
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
      void createInitMethod(std::auto_ptr<Method>& method, 
			    const MemberContainer<DataType>& args,
			    const std::string& funcName, 
			    const std::string& attName,
			    bool userInit, bool hasRetVal);
      void createUserMethod(std::auto_ptr<Method>& method, 
			    const MemberContainer<DataType>& args,
			    const std::string& funcName, 
			    const std::string& retType,
			    bool userInit, bool pureVirtual);
      DataType* _returnType;
      std::string _category;
};

#endif // Functor_H
