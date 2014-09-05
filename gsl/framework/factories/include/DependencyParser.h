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
      std::string _lensRoot;      
      std::vector<std::string> _loaded;
      bool _load(const std::string& objName);
};
#endif                           // DEPENDENCYPARSER_H
