/*
=================================================================
Licensed Materials - Property of IBM

"Restricted Materials of IBM"

BMC-YKT-03-25-2018

(C) Copyright IBM Corp. 2005-2017  All rights reserved

US Government Users Restricted Rights -
Use, duplication or disclosure restricted by
GSA ADP Schedule Contract with IBM Corp.

================================================================
*/
#ifndef __STRINGUTILS_H__
#define __STRINGUTILS_H__

#include <string>
#include <vector>
#include <sstream>
#include <string.h>
#include <fstream>

class StringUtils
{
  public:
  static void Tokenize(const std::string& str, std::vector<std::string>& tokens,
                       const std::string& delimiters = " ");
  static void wait();
  static bool fgets(char *str, int num, FILE * stream); 
  static bool streamGet(std::istringstream& istream, char* str, int num, char delim);
  static void ltrim(std::string &s);
  static void rtrim(std::string &s);
  static void trim(std::string &s);
  static inline std::string ltrimmed(std::string s) {
	  StringUtils::ltrim(s);
	  return s;
  }

  // trim from end (copying)
  static inline std::string rtrimmed(std::string s) {
	  StringUtils::rtrim(s);
	  return s;
  }

  // trim from both ends (copying)
  static inline std::string trimmed(std::string s) {
	  StringUtils::trim(s);
	  return s;
  }
  static int readOneWord(FILE* fpF, char* oneword);
  static std::string random_string( size_t length );
  -
-  /* convert an ifstream to a single string */
-  static std::string to_string(std::ifstream& in) {
-    std::stringstream sstr;
-    sstr << in.rdbuf();
-    return sstr.str();
-  }
-  //static std::string slurp(std::ifstream& in) {
-  //  return static_cast<std::stringstream const&>(std::stringstream() << in.rdbuf()).str();
-  //}
};

#endif
