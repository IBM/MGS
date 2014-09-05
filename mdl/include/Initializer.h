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
