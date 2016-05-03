#ifndef __STRINGUTILS_H__
#define __STRINGUTILS_H__

#include <string>
#include <vector>
#include <sstream>

class StringUtils
{
  public:
  static void Tokenize(const std::string& str, std::vector<std::string>& tokens,
                       const std::string& delimiters = " ");
  static void wait();
  static bool fgets(char *str, int num, FILE * stream); 
  static bool streamGet(std::istringstream& istream, char* str, int num, char delim);
};

#endif
