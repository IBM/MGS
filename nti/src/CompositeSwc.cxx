// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BMC-YKT-08-2012
//
// (C) Copyright IBM Corp. 2005-2014  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// ================================================================

#include "CompositeSwc.h"
#include <float.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iterator>
#include <cassert>
#include <vector>
#include <string>
#include <stdlib.h>
#define DEFAULT_REPLACEMENT_SOMA_TYPE 2

CompositeSwc::CompositeSwc(const char* tissueFileName, const char* outFileName, double sampleRate, bool multicolor)
{
  std::ifstream tissueFile;
  tissueFile.open(tissueFileName, std::ios::in);
  std::ofstream outFile;
  outFile.open(outFileName, std::ios::out);
  assert(outFile);
  if (tissueFile.is_open()) {
    int idx=0, root=0, n=0, nextType=1;
    double pos=0.0, stride=1.0/sampleRate;
    while ( tissueFile.good() ) {
      std::string tissueLine;
      getline (tissueFile, tissueLine);
      if (tissueLine!="" && tissueLine.at(0)!='#')  {
        if (double(n)>=pos) {
	  std::string str = tissueLine;
	  // construct a stream from the string
	  std::stringstream strstr(str);
			
	  // use stream iterators to copy the stream to the vector as whitespace separated strings
	  std::istream_iterator<std::string> it(strstr);
	  std::istream_iterator<std::string> end;
	  std::vector<std::string> tissueResults(it, end);
			
	  char* swcFileName = new char[tissueResults.at(0).length()+1];
	  strcpy(swcFileName, tissueResults.at(0).c_str());
	  std::ifstream swcFile (swcFileName);
	  root=idx;
	  if (multicolor) if (++nextType>4) nextType=2;
	  if (swcFile.is_open()) {
	    while ( swcFile.good() ) {
	      std::string swcLine;
	      getline (swcFile, swcLine);
	      if (swcLine!="" && swcLine.at(0)!='#')  {
		std::stringstream strstr2(swcLine);
		// use stream iterators to copy the stream to the vector as whitespace separated strings
		std::istream_iterator<std::string> it2(strstr2);
		std::istream_iterator<std::string> end2;
		std::vector<std::string> swcResults(it2, end2);
		int id=atoi(swcResults.at(0).c_str())+root;
		int type=0;
		if (multicolor) type=nextType; 
		else type=atoi(swcResults.at(1).c_str());
		int parent=atoi(swcResults.at(6).c_str());
		if (parent>0) parent+=root;
		else type=DEFAULT_REPLACEMENT_SOMA_TYPE;
		outFile<<id<<" "<<type<<" ";
		for (int i=2; i<6; ++i) outFile<<swcResults.at(i)<<" ";
		outFile<<parent<<std::endl;
		++idx;
	      }
	    }
	    swcFile.close();
	  }
	  pos+=stride;
	}
	++n;
      }
    }
    tissueFile.close();
    outFile.close();
  }
}
