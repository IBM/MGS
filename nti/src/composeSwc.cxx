// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "CompositeSwc.h"

#include <string>
#include <iostream>
#include <iterator>
#include <sstream>
#include <cstdlib>
#include <cstdio>

int main(int argc, char *argv[])
{
  std::string tissueFileName, compositeSwcFileName;
  double sampleRate=1.0;
  bool multicolor=false;
  for (int i = 1; i < argc; i++) {
    std::string sCurrentArg = argv[i];
    if((sCurrentArg == "-?")||(sCurrentArg == "-help")||(sCurrentArg == "-info")) {
      std::cout<<"USAGE:"<<std::endl;
      std::cout<<std::endl;
      std::cout<<"./composeSWC [-tissue tissuefile] [-sample fileSampleRate] [-multicolor]" << std::endl;
      std::cout<<"Example: ./composeSWC -tissue \"minicolumn.txt\" -sample 0.5" << std::endl;
      exit(1);
    }
    if (sCurrentArg == "-tissue") {
      tissueFileName = argv[i + 1];
      int ln=tissueFileName.length();
      compositeSwcFileName=tissueFileName;
      compositeSwcFileName.erase(ln-4, 4);
      std::ostringstream compositeSwcFileNameStream;
      compositeSwcFileNameStream<<compositeSwcFileName<<".swc";
      compositeSwcFileName=compositeSwcFileNameStream.str();
      ++i;
    }
    else if (sCurrentArg == "-sample") {
      sampleRate=atof(argv[i+1]);
      ++i;
    }
    else if (sCurrentArg == "-multicolor") {
      multicolor=true;
    }
  }

  std::cerr<<"Creating "<<compositeSwcFileName<<" ..."<<std::endl;
  CompositeSwc(tissueFileName.c_str(), compositeSwcFileName.c_str(), sampleRate, multicolor);

  exit(1);
}

