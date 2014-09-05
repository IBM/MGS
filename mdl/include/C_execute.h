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

#ifndef C_execute_H
#define C_execute_H
#include "Mdl.h"

#include "C_argumentToMemberMapper.h"
#include <memory>
#include <string>

class MdlContext;
class C_generalList;
class C_returnType;
class DataType;

class C_execute : public C_argumentToMemberMapper {

   public:
      virtual void execute(MdlContext* context);
      virtual void addToList(C_generalList* gl);
      virtual std::string getType() const;
      C_execute(C_returnType* returnType, bool ellipsisIncluded = false);
      C_execute(C_returnType* returnType, C_generalList* argumentList
		, bool ellipsisIncluded = false);
      C_execute(const C_execute& rv);
      virtual void duplicate(std::auto_ptr<C_execute>& rv) const;
      virtual void duplicate(std::auto_ptr<C_general>& rv) const;
      void releaseDataType(std::auto_ptr<DataType>& dt);
      virtual ~C_execute();

   private:
      C_returnType* _returnType;
      DataType* _dataType;
};


#endif // C_execute_H
