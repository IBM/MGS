// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef Initializer_H
#define Initializer_H
#include "Mdl.h"

#include "MemberContainer.h"
#include <string>
#include <map>
#include <vector>

class Generatable;

class Initializer
{
   public:

      Initializer(int argc, char** argv);
      ~Initializer();
      bool execute();

   private:
      void generateMakefileExtension();
      void generateCopyModules();
      void generateCopyModulesPy();

      int _argc;
      char** _argv;
      MemberContainer<Generatable>* _generatables;
      std::map<std::string, std::vector<std::string> > _copyModules;
      std::map<std::string, std::vector<std::string> > _extensionModules;
};


#endif // Initializer_H
