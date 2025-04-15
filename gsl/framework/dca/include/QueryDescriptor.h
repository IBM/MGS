// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef QUERYDESCRIPTOR_H
#define QUERYDESCRIPTOR_H
#include "Copyright.h"

#include <string>
#include <vector>
#include <memory>


class QueryField;

class QueryDescriptor
{

   private:
      std::vector<QueryField*> _fields;
      bool _anyFieldSet;

   public:
      QueryDescriptor();
      QueryDescriptor(const QueryDescriptor&);
      QueryDescriptor& operator=(const QueryDescriptor& QD);
      int addQueryField(std::unique_ptr<QueryField> & qf);
      void clearFields();
      std::vector<QueryField*> & getQueryFields();
      bool isAnyFieldSet();
      ~QueryDescriptor();
};
#endif
