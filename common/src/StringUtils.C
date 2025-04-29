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
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), 
            [](unsigned char ch) { return !std::isspace(ch); }));
}

void StringUtils::rtrim(std::string &s)
{
    s.erase(std::find_if(s.rbegin(), s.rend(), 
            [](unsigned char ch) { return !std::isspace(ch); }).base(), s.end());
}

void StringUtils::trim(std::string &s)
{
	StringUtils::ltrim(s);
	StringUtils::rtrim(s);   
}
  /**
   The word is  (in descending order of priority)
   1. fully enclosed by square bracket [   ]
   2. space-delimited 
RETURN: 
  1 = success
   */
  int StringUtils::readOneWord(FILE* fpF, char* oneword)
  {
    std::string strOneWord("");
    int errorCode = fscanf(fpF, " %s", oneword);
    unsigned int ii = 0;
    while (oneword[ii] == ' ' and ii < strlen(oneword)) {
      ii++;
    }
    unsigned int jj = strlen(oneword)-1;
    while (oneword[jj] == ' ' and jj > 0) {
      jj--;
    }
    strOneWord.append((oneword));
    if (oneword[ii] == '[') 
    {
      while (oneword[jj] != ']')
      {
	errorCode = fscanf(fpF, " %s", oneword);
	strOneWord.append((oneword));
	jj = strlen(oneword)-1;
	while (oneword[jj] == ' ' and jj > 0) {
	  jj--;
	}
      }
    }
    std::size_t length = strOneWord.copy(oneword, strOneWord.length());
    oneword[length] = '\0';
    return errorCode;
  }

std::string StringUtils::random_string( size_t length )
{
    auto randchar = []() -> char
    {
        const char charset[] =
        "0123456789"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "abcdefghijklmnopqrstuvwxyz";
        const size_t max_index = (sizeof(charset) - 1);
        return charset[ rand() % max_index ];
    };
    std::string str(length,0);
    std::generate_n( str.begin(), length, randchar );
    return str;
}

