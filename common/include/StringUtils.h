#ifndef __STRINGUTILS_H__
#define __STRINGUTILS_H__

#include <string>
#include <vector>
#include <sstream>
#include <string.h>

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
};

#endif
