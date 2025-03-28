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

#include "DependencyParser.h"
#include "LoaderException.h"
#include "LensRootConfig.h"

#ifndef DISABLE_DYNAMIC_LOADING
   #include <dlfcn.h>
#endif

#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <stdlib.h>

DependencyParser::DependencyParser(const std::string& fileName)
{
#ifndef DISABLE_DYNAMIC_LOADING
   _lensRoot = getenv("MGSROOT");
      // Temporary
   if (_lensRoot == "") {
      std::cerr << "\nMGSROOT is not set in the environment...\n"
	   << "Using " << MGSROOT << " to load shared objects.\n" << std::endl;
      _lensRoot = MGSROOT;
   }
   _fileName = _lensRoot + fileName;
#else
   _fileName = fileName;
#endif
}


DependencyParser::~DependencyParser()
{
}


bool DependencyParser::load(const std::string& objName)
{
#ifndef DISABLE_DYNAMIC_LOADING
   std::ifstream file;
   std::string token;
   std::string target = objName + ":";
   std::vector<std::string> local;
   bool found = false;
   bool result = true;

   file.open(_fileName.c_str());
   for (file >> token; file.eof() != true; file >> token) {
      if (token == target) {     // found
         found = true;
         break;
      }
   }
   if (found == false) {
      // Good place for an exception
      std::cout << "The shared object " << objName << " is not found in the Dependfile.\n"
         "Attempting to load shared object without prerequisites" << std::endl;
      return _load(objName);
   }
   found = false;
   for (file >> token; file.eof() != true; file >> token) {
      if (token == ";") {        // found
         found = true;
         break;
      }
      local.insert(local.end(), token);
      // handle the case <target>;
   }
   local.insert(local.end(), objName);
   if (found == false) {
      // Good place for an exception
      std::cout << "The targetline for " << objName << "does not end with ; in the Dependfile." << std::endl;
      return false;
   }
   std::cout << "For the shared object " << objName << ": \n";
   for (std::vector<std::string>::iterator iter = local.begin(); iter != local.end(); iter++) {
      result = result && _load(*iter);
   }
   std::cout << std::endl;
   file.close();
   return result;
#else
   return false;
#endif
}


bool DependencyParser::_load(const std::string& objName)
{
#ifndef DISABLE_DYNAMIC_LOADING
   // Search the list for previously loaded.
   for (std::vector<std::string>::iterator iter = _loaded.begin(); iter != _loaded.end(); iter++) {
      if (*iter == objName) {
         return true;
      }
   }
   std::string completeName;
   completeName = _lensRoot + "/so/" + objName + ".so";
   std::cout << "Loading " + objName + " from " + completeName + "\n";

   // attempting to load the shared object file
   void* modPtr = dlopen(completeName.c_str(), RTLD_NOW|RTLD_GLOBAL);

   // error check
   if(modPtr == 0)
      throw LoaderException("Error while opening " + completeName + ": " + dlerror());

   _loaded.insert(_loaded.end(), objName);
   return true;
#else 
   return false;
#endif
}


/*
int main(int argc, char** argv)
{
   DependencyParser dp("/home/gcaglar/xlens/so/Dependfile");
   for (int i = 1; i < argc; i++) {
      dp.load(argv[i]);
   }
   return 0;
}
*/
