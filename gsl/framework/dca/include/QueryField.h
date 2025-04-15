// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef QUERYFIELD_H
#define QUERYFIELD_H
#include "Copyright.h"

#include <string>
#include <vector>
#include <memory>


class EnumEntry;

class QueryField
{

   public:
      enum queryType{VALUE, ENUM};
      QueryField(queryType);
      QueryField(QueryField*);
      std::string getField();
      void setField(std::string field);
      std::string getName();
      void setName(std::string name);
      std::string getDescription();
      void setDescription(std::string description);
      queryType getType();
      std::vector<EnumEntry*> const & getEnumEntries() const;
      void addEnumEntry(std::unique_ptr<EnumEntry>&);
      std::string getFormat();
      void setFormat(std::string format);
      bool isSet();
      ~QueryField();

   private:
      queryType _type;
      std::string _field;
      std::string _name;
      std::string _description;
      std::vector<EnumEntry*> _enumEntries;
      std::string _format;
};
#endif
