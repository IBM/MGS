// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BCM-YKT-11-19-2015
//
// (C) Copyright IBM Corp. 2005-2015  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

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
      int addQueryField(std::auto_ptr<QueryField> & qf);
      void clearFields();
      std::vector<QueryField*> & getQueryFields();
      bool isAnyFieldSet();
      ~QueryDescriptor();
};
#endif
