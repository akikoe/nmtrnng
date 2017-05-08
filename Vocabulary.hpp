#pragma once

#include <string>
#include <unordered_map>
#include <vector>

class Vocabulary{
public:
  Vocabulary(){};
  Vocabulary(const std::string& trainFile, const int tokenFreqThreshold);
  Vocabulary(const std::string& trainFile, const int tokenFreqThreshold, const bool useSubword);
  Vocabulary(const std::string& trainFile, const bool NMTRNNG);

  class Token;

  std::unordered_map<std::string, int> tokenIndex;
  std::vector<Vocabulary::Token*> tokenList;
  int eosIndex;
  int unkIndex;
  int symIndex;
};

class Vocabulary::Token{
public:
  Token(const std::string& str_, const int count_):
    str(str_), count(count_)
  {};
  Token(const std::string& str_, const int count_, const bool subword_):
    str(str_), count(count_), subword(subword_)
  {};
  Token(const std::string& str_, const int count_, const int action_):
    str(str_), count(count_), action(action_)
  {};

  std::string str;
  int count;
  int action;
  bool subword;
};
