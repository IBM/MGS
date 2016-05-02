#ifndef __STRINGUTILS_H__
#define __STRINGUTILS_H__

#include <string>
#include <vector>

class StringUtils
{
  public:
  static void Tokenize(const std::string& str, std::vector<std::string>& tokens,
                       const std::string& delimiters = " ");
  static void wait();
};

#endif
