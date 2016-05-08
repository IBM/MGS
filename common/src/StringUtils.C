#include "../include/StringUtils.h"
#include <string>
#include <limits>
#include <iostream>
#include <cstdio>
#include <cstring>
#include <algorithm>

using namespace std;

// GOAL: parse the given string
//  into a vector of tokens
//  using the given set of delimiters
// DEFAULT: delimiter is blank space
void StringUtils::Tokenize(const string& str, vector<string>& tokens,
                           const string& delimiters)

{ /* It uses any character in the string 'delimiters'-argument as the delimiter
         */
  // Skip delimiters at beginning.
  string::size_type lastPos = str.find_first_not_of(delimiters, 0);
  // Find first "non-delimiter".
  string::size_type pos = str.find_first_of(delimiters, lastPos);

  tokens.clear(); /* make sure the vector is empty */
  while (string::npos != pos || string::npos != lastPos)
  {
    // Found a token, add it to the vector.
    tokens.push_back(str.substr(lastPos, pos - lastPos));
    // Skip delimiters.  Note the "not_of"
    lastPos = str.find_first_not_of(delimiters, pos);
    // Find next "non-delimiter"
    pos = str.find_first_of(delimiters, lastPos);
  }
}

// GOAL: just print out a message - pause the execution
// until user presses ENTER key
void StringUtils::wait()
{
  std::cout << "Press ENTER to continue...";
  std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
}

bool StringUtils::fgets(char* str, int num, FILE* stream)
{
  char* ch;
  ch = std::fgets(str, num, stream);
  bool rval = true;
  if (ch != NULL)
  {
    char* p;
    p = strchr(str, '\n');
    if (p)
    {
      // success
    }
    else
    {
      std::cerr << "Line too long, the buffer cannot hold until EOL\n";
      rval = false;
    }
  }
  else
  {
    /* fgets failed, handle error */
    std::cerr << "Reading error\n";
    rval = false;
  }

  return rval;
}

// GOAL: replicate the istringstream::get() method
//    but also add a check to ensure the buffer is long enough before reaching the 
//    delimiter
bool StringUtils::streamGet(std::istringstream & istream, char* str, int num, char delim)
{
  istream.get(str, num, delim);
  bool rval = true;
  if (istream.gcount() >= num - 1)
  {
    std::cerr << "BUFFER OVERFLOW: when reading data from a stringstream to buffer\n"
              << "either reduce line length or increase buffer size"
              << std::endl;
    rval = false;
  }
  return rval;
}

void StringUtils::ltrim(std::string &s)
{
	s.erase(s.begin(), std::find_if(s.begin(), s.end(), std::not1(std::ptr_fun<int, int>(std::isspace))));
}

void StringUtils::rtrim(std::string &s)
{
	s.erase(std::find_if(s.rbegin(), s.rend(), std::not1(std::ptr_fun<int, int>(std::isspace))).base(), s.end());
}

void StringUtils::trim(std::string &s)
{
	StringUtils::ltrim(s);
	StringUtils::rtrim(s);   
}
