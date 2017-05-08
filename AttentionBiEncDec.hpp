#pragma once

#include "LSTM.hpp"
#include "Vocabulary.hpp"
#include "SoftMax.hpp"
#include "BlackOut.hpp"
#include "Affine.hpp"

class AttentionBiEncDec{
public:

  class Data;
  class Grad;
  class DecCandidate;
  class ThreadArg;

  enum OPT{
    SGD, 
  };

  AttentionBiEncDec::OPT opt;

  AttentionBiEncDec(Vocabulary& sourceVoc_, 
		    Vocabulary& targetVoc_, 
		    std::vector<AttentionBiEncDec::Data*>& trainData_, 
		    std::vector<AttentionBiEncDec::Data*>& devData_, 
		    const int inputDim, 
		    const int hiddenEncDim, 
		    const int hiddenDim, 
		    const Real scale,
		    const bool useBlackOut_ = false, 
		    const int blackOutSampleNum = 200, 
		    const Real blackOutAlpha = 0.4,
		    const AttentionBiEncDec::OPT opt = AttentionBiEncDec::SGD, 
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
  std::vector<AttentionBiEncDec::Data*>& trainData;
  std::vector<AttentionBiEncDec::Data*>& devData;

  // Dimensional size
  int inputDim;
  int hiddenEncDim;
  int hiddenDim;

  // LSTM
  LSTM enc;
  LSTM encRev;
  LSTM dec;
  // Affine
  Affine decInitAffine;
  Affine stildeAffine;
  // Embeddings
  SoftMax softmax;
  BlackOut blackOut;
  MatD sourceEmbed;
  MatD targetEmbed;
  // Attention score
  MatD Wgeneral;

  // Initialiazed vectors
  VecD zeros;
  VecD zerosEnc;
  VecD zeros2;

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

  void biEncode(const AttentionBiEncDec::Data* data,
		AttentionBiEncDec::ThreadArg& arg, 
		const bool train);
  void biEncoderBackward2(const AttentionBiEncDec::Data* data, 
			  AttentionBiEncDec::ThreadArg& arg,
			  AttentionBiEncDec::Grad& grad);
  void decoder(AttentionBiEncDec::ThreadArg& arg,
	       std::vector<LSTM::State*>& decState, 
	       VecD& s_tilde, 
	       const int tgtNum,
	       const int i, 
	       const bool train);
  void decoderBackward2(const AttentionBiEncDec::Data* data, 
			AttentionBiEncDec::ThreadArg& arg,
			AttentionBiEncDec::Grad& grad);
  void decoderAttention(AttentionBiEncDec::ThreadArg& arg, 
			const LSTM::State* decState,
			VecD& contextSeq,
			VecD& s_tilde, 
			VecD& stildeEnd);
  void decoderAttention(AttentionBiEncDec::ThreadArg& arg, 
			const int i,
			const bool train);
  void translate(AttentionBiEncDec::Data* data, 
		 AttentionBiEncDec::ThreadArg& arg,
		 std::vector<int>& translation,
		 const bool train);
  void train(AttentionBiEncDec::Data* data, 
	     AttentionBiEncDec::ThreadArg& arg, 
	     AttentionBiEncDec::Grad& grad, 
	     const bool train);
  void calculateAlphaBiEnc(AttentionBiEncDec::ThreadArg& arg,
			   const LSTM::State* decState);
  void calculateAlphaBiEnc(AttentionBiEncDec::ThreadArg& arg,
			   const LSTM::State* decState, 
			   const int colNum);
  bool trainOpenMP(AttentionBiEncDec::Grad& grad);
  Real calcLoss(AttentionBiEncDec::Data* data, 
		AttentionBiEncDec::ThreadArg& arg, 
		const bool train = false);
  void gradientChecking(AttentionBiEncDec::Data* data, 
			AttentionBiEncDec::ThreadArg& arg,
			AttentionBiEncDec::Grad& grad);
  void gradChecker(AttentionBiEncDec::Data* data,
		   AttentionBiEncDec::ThreadArg& arg, 
		   MatD& param, 
		   const MatD& grad);
  void gradChecker(AttentionBiEncDec::Data* data,
		   AttentionBiEncDec::ThreadArg& arg, 
		   VecD& param, 
		   const MatD& grad);
  void gradChecker(AttentionBiEncDec::Data* data, 
		   AttentionBiEncDec::ThreadArg& arg, 
		   AttentionBiEncDec::Grad& grad);
  void makeTrans(const std::vector<int>& tgt, 
		 std::vector<int>& trans);
  void loadCorpus(const std::string& src, 
		  const std::string& tgt, 
		  std::vector<AttentionBiEncDec::Data*>& data);
  std::tuple<std::string, std::string> saveModel(AttentionBiEncDec::Grad& grad, 
						 const float i);
  void loadModel(AttentionBiEncDec::Grad& grad,
		 const std::string& loadModelName,
		 const std::string& loadGradName);
  void saveResult(const Real value, 
		  const std::string& name);
  static void demo(const std::string& srcTrain, 
		   const std::string& tgtTrain,
		   const std::string& srcDev, 
		   const std::string& tgtDev,
		   const int inputDim, 
		   const int hiddenEncDim, 
		   const int hiddenDim, 
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
		   const std::string& srcDev, 
		   const std::string& tgtDev,
		   const int inputDim, 
		   const int hiddenEncDim, 
		   const int hiddenDim, 
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
		       const std::string& srcDev, 
		       const std::string& tgtDev,
		       const int inputDim, 
		       const int hiddenEncDim, 
		       const int hiddenDim, 
		       const Real scale,
		       const bool useBlackOut, 
		       const int blackOutSampleNum, 
		       const Real blackOutAlpha, 
		       const int beamSize, 
		       const int maxLen, 
		       const int miniBatchSize, 
		       const int threadNum, 
		       const Real learningRate,
		       const int srcVocaThreshold, 
		       const int tgtVocaThreshold,
		       const bool istTest,
		       const std::string& saveDirName, 
		       const std::string& loadModelName, 
		       const std::string& loadGradName, 
		       const int startIter);
  void save(const std::string& fileName);
  void load(const std::string& fileName);
};

class AttentionBiEncDec::Data{
public:
  std::vector<int> src;
  std::vector<int> tgt;
  std::vector<int> trans; // Output of Decoder
};

class AttentionBiEncDec::DecCandidate{
public:
  Real score;
  std::vector<int> generatedTarget;
  LSTM::State prevDec;
  LSTM::State curDec;
  std::vector<LSTM::State*> decState;
  VecD s_tilde;
  VecD stildeEnd;
  VecD contextSeq;
  VecD targetDist;
  MatD showAlphaSeq;
  bool stop;

  DecCandidate() {};
  void init(const int maxLen = 0);
};

class AttentionBiEncDec::ThreadArg{
public:
  Rand rnd;
  // Encoder-Decoder
  std::vector<LSTM::State*> encState;
  std::vector<LSTM::State*> encRevState;
  std::vector<VecD> biEncState; // for encState and encRevState
  std::vector<LSTM::State*> decState;
  // The others
  std::vector<VecD> s_tilde;
  std::vector<VecD> contextSeqList;
  std::vector<VecD> showAlphaSeq;
  std::vector<VecD> del_stilde;
  VecD del_contextSeq;
  // Affine
  VecD decInitStateEnd;
  std::vector<VecD> stildeEnd;
  VecD del_decInitStateEnd;
  VecD del_stildeEnd;
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

  int srcLen; // srcLen
  int tgtLen; // tgtLen
  Real loss;

  std::vector<AttentionBiEncDec::DecCandidate> candidate; // for Beam Search

  ThreadArg() {};
  ThreadArg(AttentionBiEncDec& attentionBiEncDec);
  void initTrans(const int beamSize, 
		 const int maxLen);
  void initLoss();
  void init(AttentionBiEncDec& attentionBiEncDec, 
	    const AttentionBiEncDec::Data* data, 
	    const bool train = false);
};

class AttentionBiEncDec::Grad{

public:
  AttentionBiEncDec::Grad* gradHist;

  MatD sourceEmbedMatGrad;
  MatD targetEmbedMatGrad;
  MatD WgeneralMatGrad;
  std::unordered_map<int, VecD> sourceEmbed;
  std::unordered_map<int, VecD> targetEmbed;

  // LSTM
  LSTM::Grad lstmSrcGrad;
  LSTM::Grad lstmSrcRevGrad;
  LSTM::Grad lstmTgtGrad;
  
  // Affine
  Affine::Grad decInitAffineGrad;
  Affine::Grad stildeAffineGrad;
  
  SoftMax::Grad softmaxGrad;
  BlackOut::Grad blackOutGrad;

  MatD Wgeneral; // attenType = AttentionBiEncDec::GENERAL
 
  Grad(): gradHist(0) {}
  Grad(AttentionBiEncDec& attentionBiEncDec);

  void init();
  Real norm();
  void operator += (const AttentionBiEncDec::Grad& grad);
  void sgd(AttentionBiEncDec& attentionBiEncDec, 
	   const Real learningRate);
  void save(AttentionBiEncDec& attentionBiEncDec, 
	    const std::string& filename);
  void load(AttentionBiEncDec& attentionBiEncDec, 
	    const std::string& filename);
};
