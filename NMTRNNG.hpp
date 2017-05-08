#pragma once

#include <stack>
#include <tuple>
#include "LSTM.hpp"
#include "Affine.hpp"
#include "Vocabulary.hpp"
#include "SoftMax.hpp"
#include "BlackOut.hpp"

class NMTRNNG{
public:

  class Data;
  class Grad;
  class DecCandidate;
  class ThreadArg;

  enum OPT{
    SGD, 
  };
  
  NMTRNNG::OPT opt;

  NMTRNNG(Vocabulary& sourceVoc_, 
	  Vocabulary& targetVoc_, 
	  Vocabulary& actionVoc_,
	  std::vector<NMTRNNG::Data*>& trainData_, 
	  std::vector<NMTRNNG::Data*>& devData_, 
	  const int inputDim, 
	  const int inputActDim, 
	  const int hiddenEncDim, 
	  const int hiddenDim, 
	  const int hiddenActDim, 
	  const Real scale,
	  const bool useBlackOut_ = false, 
	  const int blackOutSampleNum = 10, 
	  const Real blackOutAlpha = 0.4,
	  const NMTRNNG::OPT opt = NMTRNNG::SGD, 
	  const Real clipThreshold = 3.0,
	  const int beamSize = 20, 
	  const int maxLen = 100,
	  const int miniBatchSize = 20, 
	  const int threadNum = 8,
	  const Real learningRate = 0.5, 
	  const bool isTest = false,
	  const int startIter = 0,
	  const std::string& saveDirName = "./"); // Set a path to a directory to save a model

  // Corpus
  Vocabulary& sourceVoc;
  Vocabulary& targetVoc;
  Vocabulary& actionVoc;
  std::vector<NMTRNNG::Data*>& trainData;
  std::vector<NMTRNNG::Data*>& devData;

  // Dimensional size
  int inputDim;
  int inputActDim;
  int hiddenEncDim;
  int hiddenDim;
  int hiddenActDim;

  // LSTM units
  LSTM enc;
  LSTM encRev;
  LSTM dec;
  LSTM act; // RNNG 
  LSTM outBuf; // RNNG's Stack
  // Affine
  Affine decInitAffine;
  Affine actInitAffine;
  Affine outBufInitAffine;
  Affine utAffine;
  Affine stildeAffine;
  Affine embedVecAffine;
  // Embeddings
  SoftMax softmax;
  SoftMax softmaxAct;
  BlackOut blackOut;
  MatD sourceEmbed;
  MatD targetEmbed;
  MatD actionEmbed;
  // Initialized vectors
  VecD zeros;
  VecD zerosEnc;
  VecD zeros2;
  VecD zerosAct;
  // Attention score
  MatD Wgeneral;

  bool useBlackOut;
  Real clipThreshold;
  Rand rnd;

  int beamSize;
  int maxLen;
  int miniBatchSize;
  int threadNum;
  Real learningRate;
  bool isTest;
  int startIter;
  std::string saveDirName;

  /* for automated tuning */
  Real prevPerp;
  std::string prevModelFileName;
  std::string prevGradFileName;

  void biEncode(const NMTRNNG::Data* data, 
		NMTRNNG::ThreadArg& arg, 
		const bool train = false);
  void biEncoderBackward2(const NMTRNNG::Data* data, 
			  NMTRNNG::ThreadArg& arg, 
			  NMTRNNG::Grad& grad);
  void decoder(NMTRNNG::ThreadArg& arg, 
	       std::vector<LSTM::State*>& decState, 
	       VecD& s_tilde, 
	       const int tgtNum, 
	       const int i, 
	       const bool train = false);
  void decoderActionBackward2(const NMTRNNG::Data* data, 
			      NMTRNNG::ThreadArg& arg, 
			      NMTRNNG::Grad& grad);
  void decoderBackward2(const NMTRNNG::Data* data, 
			NMTRNNG::ThreadArg& arg, 
			NMTRNNG::Grad& grad);
  void decoderAction(NMTRNNG::ThreadArg& arg,
		     std::vector<LSTM::State*>& actState, 
		     const int actNum,
		     const int i, 
		     const bool train = false);
  void candidateDecoder(NMTRNNG::ThreadArg& arg, 
			std::vector<LSTM::State*>& decState,
			const VecD& s_tilde, 
			const std::vector<int>& tgt, 
			const int i);
  void decoderReduceLeft(NMTRNNG::Data* data, 
			 NMTRNNG::ThreadArg& arg, 
			 const int phraseNum, 
			 const int actNum, 
			 const int k,
			 const bool = true);
  void decoderReduceRight(NMTRNNG::Data* data, 
			  NMTRNNG::ThreadArg& arg, 
			  const int phraseNum, 
			  const int actNum, 
			  const int k,
			  const bool = true);
  void decoderReduceLeftCand(NMTRNNG::ThreadArg& arg, 
			     std::vector<LSTM::State*>& outBufState,
			     NMTRNNG::DecCandidate& cand, 
			     std::vector<int>& tgt,
			     const int actNum, 
			     const int k, 
			     const bool train);
  void decoderReduceRightCand(NMTRNNG::ThreadArg& arg, 
			      std::vector<LSTM::State*>& outBufState, 
			      NMTRNNG::DecCandidate& cand, 
			      std::vector<int>& tgt,
			      const int actNum, 
			      const int k, 
			      const bool train);
  void compositionFunc(VecD& c, 
		       const VecD& head, 
		       const VecD& dependent, 
		       const VecD& relation, 
		       VecD& embedVecEnd);
  void reduceHeadStack(std::stack<int>& stack, 
		       int& top, 
		       const int k);
  void reduceStack(std::stack<int>& stack, 
		   int& right, 
		   int& left);
  void decoderAttention(NMTRNNG::ThreadArg& arg, 
			const LSTM::State* decState, 
			VecD& contextSeq, 
			VecD& s_tilde, 
			VecD& stildeEnd);
  void decoderAttention(NMTRNNG::ThreadArg& arg, 
			const int i,
			const bool train);
  void translate(NMTRNNG::Data* data, 
		 NMTRNNG::ThreadArg& arg, 
		 std::vector<int>& translation, 
		 const bool train = false);
  void translateWithAction(NMTRNNG::Data* data, 
			   NMTRNNG::ThreadArg& arg, 
			   std::vector<int>& translation, 
			   const int beamSizeA, 
			   const bool train = false);
  void translateWithStat(NMTRNNG::Data* data, 
			 NMTRNNG::ThreadArg& arg,
			 const std::unordered_map<int, std::unordered_map<int, Real> >& stat,
			 const bool greedy = false);
  void train(NMTRNNG::Data* data, 
	     NMTRNNG::ThreadArg& arg, 
	     NMTRNNG::Grad& grad, 
	     const bool train);
  void sgd(const NMTRNNG::Grad& grad, 
	   const Real learningRate);
  void train();
  bool trainOpenMP(NMTRNNG::Grad& grad);
  void calculateAlpha(NMTRNNG::ThreadArg& arg, 
		      const LSTM::State* decState);
  void calculateAlpha(NMTRNNG::ThreadArg& arg, 
		      const LSTM::State* decState, 
		      const int colNum);
  std::tuple<Real, Real> calcLoss(NMTRNNG::Data* data, 
				  NMTRNNG::ThreadArg& arg, 
				  const bool train = false);
  void gradientChecking(NMTRNNG::Data* data, 
			NMTRNNG::ThreadArg& arg,
			NMTRNNG::Grad& grad);
  void gradChecker(NMTRNNG::Data* data,
		   NMTRNNG::ThreadArg& arg,
		   MatD& param, 
		   const MatD& grad);
  void gradChecker(NMTRNNG::Data* data,
		   NMTRNNG::ThreadArg& arg, 
		   VecD& param, 
		   const MatD& grad);
  void gradChecker(NMTRNNG::Data* data, 
		   NMTRNNG::ThreadArg& arg,
		   NMTRNNG::Grad& grad);
  void makeTrans(const std::vector<int>& tgt, 
		 std::vector<int>& trans);
  void loadCorpus(const std::string& src, 
		  const std::string& tgt, 
		  const std::string& act,
		  std::vector<NMTRNNG::Data*>& data);
  void save(const std::string& fileName);
  void load(const std::string& fileName);
  std::tuple<std::string, std::string> saveModel(NMTRNNG::Grad& grad, 
						 const float i);
  void loadModel(NMTRNNG::Grad& grad, 
		 const std::string& loadModelName, 
		 const std::string& loadGradName);
  void saveResult(const Real value, 
		  const std::string& name);
  static void demo(const std::string& srcTrain, 
		   const std::string& tgtTrain, 
		   const std::string& actTrain,
		   const std::string& srcDev, 
		   const std::string& tgtDev, 
		   const std::string& actDev,
		   const int inputDim, 
		   const int inputActDim, 
		   const int hiddenEncDim, 
		   const int hiddenDim, 
		   const int hiddenActDim,
		   const Real scale,
		   const bool useBlackOut, 
		   const int blackOutSampleNum, 
		   const Real blackOutAlpha, 
		   const Real clipThreshold,
		   const int beamSize, 
		   const int maxLen, 
		   const int miniBatchSize, 
		   const int threadNum, 
		   const Real learningRate,
		   const int srcVocaThreshold, 
		   const int tgtVocaThreshold,
		   const std::string& saveDirName);
  static void demo(const std::string& srcTrain, 
		   const std::string& tgtTrain, 
		   const std::string& actTrain,
		   const std::string& srcDev, 
		   const std::string& tgtDev, 
		   const std::string& actDev,
		   const int inputDim, 
		   const int inputActDim, 
		   const int hiddenEncDim, 
		   const int hiddenDim, 
		   const int hiddenActDim, 
		   const Real scale,
		   const bool useBlackOut, 
		   const int blackOutSampleNum, 
		   const Real blackOutAlpha, 
		   const Real clipThreshold,
		   const int beamSize, 
		   const int maxLen, 
		   const int miniBatchSize, 
		   const int threadNum, 
		   const Real learningRate,
		   const int srcVocaThreshold, 
		   const int tgtVocaThreshold,
		   const std::string& saveDirName, 
		   const std::string& loadModelName, 
		   const std::string& loadGradName, 
		   const int startIter);
  static void evaluate(const std::string& srcTrain, 
		       const std::string& tgtTrain, 
		       const std::string& actTrain,
		       const std::string& srcDev, 
		       const std::string& tgtDev, 
		       const std::string& actDev,
		       const int inputDim, 
		       const int inputActDim, 
		       const int hiddenEncDim, 
		       const int hiddenDim, 
		       const int hiddenActDim, 
		       const Real scale,
		       const bool useBlackOut, 
		       const int blackOutSampleNum, 
		       const Real blackOutAlpha, 
		       const int beamSize, 
		       const int maxGeneNum, 
		       const int miniBatchSize, 
		       const int threadNum, 
		       const Real learningRate,
		       const int srcVocaThreshold, 
		       const int tgtVocaThreshold,
		       const bool isTest, 
		       const std::string& saveDirName, 
		       const std::string& loadModelName, 
		       const std::string& loadGradName, 
		       const int startIter);
};

class NMTRNNG::Data{
public:
  std::vector<int> src;
  std::vector<int> tgt;
  std::vector<int> action;
  std::vector<int> trans; // Output of Decoder
};

class NMTRNNG::DecCandidate{
public:
  Real score;
  Real scoreAct;
  std::vector<int> generatedTarget;
  std::vector<int> generatedAction;
  LSTM::State prevDec;
  LSTM::State curDec;
  LSTM::State prevAct;
  LSTM::State curAct;
  std::vector<LSTM::State*> decState;
  std::vector<LSTM::State*> actState;
  std::vector<LSTM::State*> outBufState;
  std::vector<VecD> embedVecEnd;
  VecD decInitStateEnd;
  VecD s_tilde;
  VecD stildeEnd;
  VecD contextSeq;
  VecD ut;
  VecD utEnd;
  VecD targetDist;
  VecD targetActDist;
  MatD showAlphaSeq;
  bool stop;
  int i, k;
  int phraseNum;
  int tgtLen;

  std::stack<int> headStack;
  std::vector<int> headList; // head's history
  std::stack<int> embedStack;
  std::vector<int> embedList;
  std::vector<VecD> embedVec;

  DecCandidate() {};
  void init(NMTRNNG& nmtRNNG);
};

class NMTRNNG::ThreadArg{
public:
  Rand rnd;
  // Encoder-Decoder
  std::vector<LSTM::State*> encState;
  std::vector<LSTM::State*> encRevState;
  std::vector<VecD> biEncState; // for encState and encRevState
  std::vector<LSTM::State*> decState;
  std::vector<LSTM::State*> actState; // RNNG's Action
  std::vector<LSTM::State*> outBufState; // RNNG's 
  // The others
  std::vector<VecD> s_tilde;
  std::vector<VecD> ut;
  std::vector<VecD> embedVec;
  std::vector<VecD> contextSeqList;
  std::vector<VecD> showAlphaSeq;
  std::vector<VecD> del_stilde; // decoder and its gradient for input-feeding
  std::vector<VecD> del_ut;
  std::unordered_map<int, VecD> del_embedVec;
  VecD del_contextSeq;
  // Affine
  VecD decInitStateEnd;
  VecD encStateEnd;
  VecD outBufInitStateEnd;
  std::vector<VecD> stildeEnd;
  std::vector<VecD> utEnd;
  std::vector<VecD> embedVecEnd;
  VecD del_decInitStateEnd;
  VecD del_encStateEnd;
  VecD del_outBufInitStateEnd;
  VecD del_stildeEnd;
  VecD del_utEnd;
  VecD del_embedVecEnd;
  // Attention Score
  MatD alphaSeq;
  VecD alphaSeqVec;
  VecD del_alphaSeq;
  VecD del_alphaSeqTmp;
  VecD del_alignScore;
  VecD del_WgeneralTmp;

  std::vector<BlackOut::State> blackOutState;
  std::vector<VecD> targetDistVec;
  VecD targetDist;
  VecD actionDist;

  std::stack<int> headStack;
  std::vector<int> headList; // head's history
  std::stack<int> embedStack;
  std::vector<int> embedList;

  int srcLen; // srcLen
  int tgtLen; // tgtLen
  int actLen; // actLen
  Real loss;
  Real lossAct;

  std::vector<NMTRNNG::DecCandidate> candidate; // for Beam Search

  ThreadArg () {};
  ThreadArg(NMTRNNG& nmtRNNG);
  void initTrans(NMTRNNG& nmtRNNG, 
		 const int beamSize);
  void clear();
  void initLoss();
  void init(NMTRNNG& nmtRNNG, 
	    const NMTRNNG::Data* data, 
	    const bool train = false);
};

class NMTRNNG::Grad{
 
public:
  NMTRNNG::Grad* gradHist;

  MatD sourceEmbedMatGrad;
  MatD targetEmbedMatGrad;
  MatD actionEmbedMatGrad;
  MatD WgeneralMatGrad;
  std::unordered_map<int, VecD> sourceEmbed;
  std::unordered_map<int, VecD> targetEmbed;
  std::unordered_map<int, VecD> actionEmbed;

  // LSTM
  LSTM::Grad lstmSrcGrad;
  LSTM::Grad lstmSrcRevGrad;
  LSTM::Grad lstmTgtGrad;
  LSTM::Grad lstmActGrad;
  LSTM::Grad lstmOutBufGrad;
  
  // Affine
  Affine::Grad decInitAffineGrad;
  Affine::Grad actInitAffineGrad;
  Affine::Grad outBufInitAffineGrad;
  Affine::Grad utAffineGrad;
  Affine::Grad stildeAffineGrad;
  Affine::Grad embedVecAffineGrad;

  SoftMax::Grad softmaxGrad;
  SoftMax::Grad softmaxActGrad;
  BlackOut::Grad blackOutGrad;

  MatD Wgeneral; // attenType = NMTRNNG::GENERAL
  
  Grad(): gradHist(0) {}
  Grad(NMTRNNG& nmtRNNG);
 
  void init();
  Real norm();
  void operator += (const NMTRNNG::Grad& grad);
  void sgd(NMTRNNG& nmtRNNG, 
	   const Real learningRate);
  void save(NMTRNNG& nmtRNNG, 
	    const std::string& filename);
  void load(NMTRNNG& nmtRNNG, 
	    const std::string& filename);
};
