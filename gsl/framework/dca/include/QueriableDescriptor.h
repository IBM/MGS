// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef QUERIABLEDESCRIPTOR_H
#define QUERIABLEDESCRIPTOR_H
#include "Copyright.h"

#include <string>


class Queriable;

class QueriableDescriptor
{

   public:

      QueriableDescriptor();
      QueriableDescriptor(QueriableDescriptor&);
      QueriableDescriptor& operator=(const QueriableDescriptor& QD);
      std::string getName();
      void setName(std::string name);
      std::string getDescription();
      void setDescription(std::string description);
      std::string getType();
      void setType(std::string type);
      Queriable* getQueriable();
      void setQueriable(Queriable* ptrQueriable);
      ~QueriableDescriptor();

   private:

      std::string _name;
      std::string _description;
      std::string _type;
      Queriable* _ptrQueriable;
};
#endif
