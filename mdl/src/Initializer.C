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

#include <memory>
#include <iostream>
#include <fstream>
#include <sstream>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>

#include "Initializer.h"
#include "MdlLexer.h"
#include "MdlContext.h"
#include "CommandLine.h"
#include "MemberContainer.h"
#include "Generatable.h"
#include "GeneralException.h"
#include "InternalException.h"
#include "Constants.h"

extern int mdlparse(void*);
#if YYDEBUG
extern int yydebug;
#endif

Initializer::Initializer(int argc, char** argv)
   : _argc(argc), _argv(argv)
{
}

bool Initializer::execute()
{
   CommandLine commandLine;
   std::auto_ptr<MdlLexer> scanner;
   std::auto_ptr<MdlContext> context;

   if (!commandLine.parse(_argc, _argv)) {
      return false;
   }

   std::string fileName = commandLine.getFileName();
   char const * infilename = fileName.c_str();
   std::istream *infile;
   std::ostream *outfile = &std::cout;
#if YYDEBUG
   yydebug = 0;
#endif
   char temporaryName[256] = "/tmp/bc_mpp.XXXXXX";             // modified by Jizhu Lu on 01/10/2006

   std::ostringstream command;
//   if (tmpnam(temporaryName)) {                         // commented out by Jizhu Lu on 01/10/2006
   if (mkstemp(temporaryName)) {                          // added by Jizhu Lu on 01/10/2006
      #ifdef LINUX
      command << "cpp ";
      #endif
      #ifdef AIX
      command << "/usr/gnu/bin/gcpp ";
      #endif
      command << " " << infilename << " " << temporaryName;

      const std::vector<std::string>& includePath = 
	 commandLine.getIncludePath();
      std::vector<std::string>::const_iterator it, 
	 end = includePath.end();
      for (it = includePath.begin(); it != end; ++it) {
	 command << " -I" << *it;
      }

      system(command.str().c_str());
      infile = new std::ifstream(temporaryName);
   }
   else {
      std::cerr << "Unable to open temporary file, cpp problem, aborting..." 
		<< std::endl;
      return false;
   }

   scanner.reset(new MdlLexer(infile,outfile));
   context.reset(new MdlContext());
   context->_lexer = scanner.get();
   _generatables = new MemberContainer<Generatable>();
   context->_generatables = _generatables;
   try {
      mdlparse(context.get());
   } catch (InternalException& e) {
      std::cerr << "Internal error: " << e.getError() << std::endl;
      std::cerr << "Quitting..." << std::endl;      
      return false;
   } catch (GeneralException& e) {
      if (e.getError() != "") {
	 std::cerr << "Uncaught exception at main level: " << e.getError() 
		   << std::endl;
      }
      std::cerr << "Quitting..." << std::endl;
      return false;
   } catch (...) {
      std::cerr << "Uncaught exception at main level." << std::endl;	 
      std::cerr << "Quitting..." << std::endl;
      return false;
   }

   // If there was an error return /wo starting...
   if (context->isError()) {
      std::cerr << "Quitting due to errors..." << std::endl;
      return false;
   }

   MemberContainer<Generatable>::const_iterator end = 
      context->_generatables->end();
   MemberContainer<Generatable>::const_iterator it;
   try {
      for (it = context->_generatables->begin(); it != end; it++) {
	 it->second->generate();
      }
   } catch (InternalException& e) {
      std::cerr << "Internal error: " << e.getError() << std::endl;
      std::cerr << "Quitting..." << std::endl;
      return false;
   }

   if (commandLine.isStatic()) {
      for (it = context->_generatables->begin(); it != end; it++) {
	 it->second->setLinkType(Generatable::_STATIC);
      }
   }

   try {
      for (it = context->_generatables->begin(); it != end; it++) {
	 it->second->generateFiles(fileName);
      }
   } catch (InternalException& e) {
      std::cerr << "Internal error: " << e.getError() << std::endl;
      std::cerr << "Quitting..." << std::endl;      
      return false;
   } catch (...) {
      std::cerr << "Quitting..." << std::endl;      
      return false;
   }

   for (it = context->_generatables->begin(); it != end; it++) {
      it->second->addSelfToExtensionModules(_extensionModules);
      it->second->addSelfToCopyModules(_copyModules);
   }
   generateMakefileExtension(); // hack for MBL
   generateCopyModules(); // hack for MBL
//   generateCopyModulesPy();
   delete infile;
   unlink(temporaryName);                      // added by Jizhu Lu on 01/10/2006 to remove the temporary file.
   return true;
}

void Initializer::generateMakefileExtension()
{   
  std::ostringstream os;

  std::map<std::string, std::vector<std::string> >::iterator it, 
    end = _extensionModules.end();
  for(it = _extensionModules.begin(); it != end; ++it) {
    std::string type = it->first;
    for(unsigned int i = 0; i < type.size(); ++i) {
      if ((type[i] >= 'a') && (type[i] <= 'z')) {
        type[i] += 'A' - 'a';
      }
    }

    os << "#{{{" << type << "\n";
    //std::endl;
    os << type << "_MODULES += ";
    std::vector<std::string>::iterator it2, end2 = it->second.end();
    for (it2 = it->second.begin(); it2 != end2; ++it2) {
      if (it2 != it->second.begin()) {
        os << "\t";
      }
      os << *it2 << " \\\n";
    }
    os << "#}}}\n";
    os << "\n";
  }   
  std::ofstream fs("Extensions.mk");
  if (fs.is_open())
  {
    fs << os.str();
    fs.close();
  }
  else
  {
    std::cerr << "Cannot create/open Extensions.mk file" << std::endl;
    assert(0);
  }
}

void Initializer::generateCopyModules()
{
   std::ostringstream os;
   
   std::map<std::string, std::vector<std::string> >::iterator it, 
      end = _copyModules.end();
   for(it = _copyModules.begin(); it != end; ++it) {
     os << "mkdir -p $LENSROOT/extensions/" << it->first << "\n";
     os << "cp -r ";
     std::vector<std::string>::iterator it2, end2 = it->second.end();
     for (it2 = it->second.begin(); it2 != end2; ++it2) {
       if (it2 != it->second.begin()) {
         os << " ";
       }
       os << *it2 << " ";
     }
     os << " $LENSROOT/extensions/" << it->first << "> /dev/null 2>&1 || : \n";
   }   
   //TUAN: no need to overwrite Extensions.mk if we call mdlparser directly
   // this can be done (if needed), by calling via the ../define script
   //os << "touch $LENSROOT/Extensions.mk\n";
   //os << "cp $LENSROOT/Extensions.mk $LENSROOT/Extensions.mk.bak\n";
   //os << "cp Extensions.mk $LENSROOT/\n";
   std::ofstream fs("copyModules");
   if (fs.is_open())
   {
     fs << os.str();
     fs.close();
     system("sh copyModules");
   }
   else
   {
     std::cerr << "Cannot create/open copyModules file" << std::endl;
     assert(0);
   }
}

void Initializer::generateCopyModulesPy()
{
   std::ostringstream os;
   
   // The fixed body of the string
   os << "#!/usr/bin/python\n"
      << "\n"
      << "modules = [";

   std::map<std::string, std::vector<std::string> >::iterator it, 
      end = _copyModules.end();
   for(it = _copyModules.begin(); it != end; ++it) {
      std::vector<std::string>::iterator it2, end2 = it->second.end();
      for (it2 = it->second.begin(); it2 != end2; ++it2) {
	 if ((it != _copyModules.begin()) || (it2 != it->second.begin())) {
	    os << ", ";
	 }
	 os << "(\"" << it->first << "\",\"" << *it2 << "\")";
      }
   }   
   os << "]\n"
      << "\n"
      << "for module in modules:\n"
      << TAB << "print module[0], module[1]\n";

   std::string fileName = "copyModules.py";
   std::ofstream fs(fileName.c_str());
   fs << os.str();
   fs.close();
   std::string cmd = "chmod +x ";
   cmd += fileName;
   system(cmd.c_str());
}
