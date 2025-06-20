// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef DEPENDENCYPARSER_H
#define DEPENDENCYPARSER_H
#include "Copyright.h"
//#include <fstream>
#include<string>
#include<vector>

class DependencyParser
{
   public:
      DependencyParser(const std::string& fileName);
      ~DependencyParser();
      bool load(const std::string& objName);
   private:
      std::string _fileName;
      std::string _gslRoot;      
      std::vector<std::string> _loaded;
      bool _load(const std::string& objName);
};
#endif                           // DEPENDENCYPARSER_H
