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

#ifndef C_initialize_H
#define C_initialize_H
#include "Mdl.h"

#include "C_argumentToMemberMapper.h"
#include <memory>
#include <string>

class MdlContext;
class ToolBase;
class C_generalList;

class C_initialize : public C_argumentToMemberMapper {

   public:
      virtual void execute(MdlContext* context);
      virtual void addToList(C_generalList* gl);
      virtual std::string getType() const;
      C_initialize(bool ellipsisIncluded = false);
      C_initialize(C_generalList* argumentList, bool ellipsisIncluded = false);
      C_initialize(const C_initialize& rv);
      virtual void duplicate(std::auto_ptr<C_initialize>& rv) const;
      virtual void duplicate(std::auto_ptr<C_general>& rv) const;
      virtual ~C_initialize();
};


#endif // C_initialize_H
