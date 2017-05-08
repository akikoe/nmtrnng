#include "NMTRNNG.hpp"
#include "ActFunc.hpp"
#include "Optimizer.hpp"
#include "Utils.hpp"
#include "Affine.hpp"
#include <pthread.h>
#include <sys/time.h>
#include <omp.h>
#include <chrono>
#include <stack>
#include <tuple>
#include <iostream>
#include <fstream>

/* NMT + RNNG + Transition-based Dependency parser:
   Paper: "Learning to Parse and Translate Improves Neural Machine Translation"
   Pdf: https://arxiv.org/pdf/1702.03525.pdf
   
   Encoder: 1-layer bi-directional encoder
   + Speedup technique proposed by Yuchen Qiao
   + biEncHalf
   + Optimizer (SGD)

   References
   Paper: "Recurrent Neural Network Grammar"
   Pdf: https://arxiv.org/pdf/1602.07776v4.pdf

   Paper: "Transition-Based Dependency Parsing with Stack Long Short-Term Memory"
   Pdf: https://arxiv.org/pdf/1505.08075v1.pdf

   Paper: "Cache Friendly Parallelization of Neural Encoder-Decoder Models without Padding on Multi-core Architecture"
   Pdf: to appar at ParLearning 2017

*/

#define print(var)  \
  std::cout<<(var)<<std::endl

NMTRNNG::NMTRNNG(Vocabulary& sourceVoc_, 
		 Vocabulary& targetVoc_, 
		 Vocabulary& actionVoc_, 
		 std::vector<NMTRNNG::Data*>& trainData_, 
		 std::vector<NMTRNNG::Data*>& devData_, 
		 const int inputDim_, 
		 const int inputActDim_, 
		 const int hiddenEncDim_, 
		 const int hiddenDim_, 
		 const int hiddenActDim_, 
		 const Real scale,
		 const bool useBlackOut_, 
		 const int blackOutSampleNum, 
		 const Real blackOutAlpha,
		 const NMTRNNG::OPT opt_,
		 const Real clipThreshold_,
		 const int beamSize_, 
		 const int maxLen_,
		 const int miniBatchSize_, 
		 const int threadNum_, 
		 const Real learningRate_, 
		 const bool isTest_,
		 const int startIter_,
		 const std::string& saveDirName_):
opt(opt_),
  sourceVoc(sourceVoc_), 
  targetVoc(targetVoc_), 
  actionVoc(actionVoc_),
  trainData(trainData_), 
  devData(devData_),
  inputDim(inputDim_), 
  inputActDim(inputActDim_), 
  hiddenEncDim(hiddenEncDim_),
  hiddenDim(hiddenDim_), 
  hiddenActDim(hiddenActDim_), 
  useBlackOut(useBlackOut_),
  clipThreshold(clipThreshold_),
  beamSize(beamSize_), 
  maxLen(maxLen_), 
  miniBatchSize(miniBatchSize_), 
  threadNum(threadNum_), 
  learningRate(learningRate_), 
  isTest(isTest_),
  startIter(startIter_),
  saveDirName(saveDirName_)
{
  // this->rnd = Rand(this->rnd.next()); // (!) TODO: For Ensemble

  // LSTM units
  this->enc = LSTM(inputDim, hiddenEncDim); // Encoder; set dimension
  this->enc.init(this->rnd, scale);
  this->encRev = LSTM(inputDim, hiddenEncDim); // Encoder-Reverse; set dimension
  this->encRev.init(this->rnd, scale);

  this->dec = LSTM(inputDim, hiddenDim, hiddenDim); // Decoder; set dimension
  this->dec.init(this->rnd, scale);
  this->act = LSTM(inputActDim, hiddenActDim);
  this->act.init(this->rnd, scale);
  this->outBuf = LSTM(inputDim, hiddenDim);
  this->outBuf.init(this->rnd, scale);

  // LSTMs' biases set to 1 
  this->enc.bf.fill(1.0);
  this->encRev.bf.fill(1.0);

  this->dec.bf.fill(1.0);
  this->act.bf.fill(1.0);
  this->outBuf.bf.fill(1.0);

  // Affine
  this->decInitAffine = Affine(hiddenEncDim*2, hiddenDim);
  this->decInitAffine.act = Affine::TANH;
  this->decInitAffine.init(this->rnd, scale);
  this->actInitAffine = Affine(hiddenEncDim*2, hiddenActDim);
  this->actInitAffine.act = Affine::TANH;
  this->actInitAffine.init(this->rnd, scale);
  this->outBufInitAffine = Affine(hiddenEncDim*2, hiddenDim);
  this->outBufInitAffine.act = Affine::TANH;
  this->outBufInitAffine.init(this->rnd, scale);

  this->utAffine = Affine(hiddenDim*2 + hiddenActDim, hiddenDim);
  this->utAffine.act = Affine::TANH;
  this->utAffine.init(this->rnd, scale);
  this->stildeAffine = Affine(hiddenDim + hiddenEncDim*2, hiddenDim);
  this->stildeAffine.act = Affine::TANH;
  this->stildeAffine.init(this->rnd, scale);
  this->embedVecAffine = Affine(inputDim*2 + inputActDim, inputDim);
  this->embedVecAffine.act = Affine::TANH;
  this->embedVecAffine.init(this->rnd, scale);

  // Embedding matrices
  this->sourceEmbed = MatD(inputDim, this->sourceVoc.tokenList.size());
  this->targetEmbed = MatD(inputDim, this->targetVoc.tokenList.size());
  this->actionEmbed = MatD(inputActDim, this->actionVoc.tokenList.size());
  this->rnd.uniform(this->sourceEmbed, scale);
  this->rnd.uniform(this->targetEmbed, scale);
  this->rnd.uniform(this->actionEmbed, scale);

  this->Wgeneral = MatD(hiddenDim, hiddenEncDim*2); // attenType == GENERAL; for Sequence
  this->rnd.uniform(this->Wgeneral, scale);

  // Softmax / BlackOut
  if (!this->useBlackOut) {
    this->softmax = SoftMax(hiddenDim, this->targetVoc.tokenList.size());
  } else {
    VecD freq = VecD(this->targetVoc.tokenList.size());
    for (int i = 0; i < (int)this->targetVoc.tokenList.size(); ++i) {
      freq.coeffRef(i, 0) = this->targetVoc.tokenList[i]->count;
    }
    this->blackOut = BlackOut(hiddenDim, this->targetVoc.tokenList.size(), blackOutSampleNum);
    this->blackOut.initSampling(freq, blackOutAlpha);
  }
  this->softmaxAct = SoftMax(hiddenDim, this->actionVoc.tokenList.size());

  // The others
  this->zeros = VecD::Zero(hiddenDim); // Zero vector
  this->zerosEnc = VecD::Zero(this->hiddenEncDim); // Zero vector
  this->zeros2 = VecD::Zero(this->hiddenEncDim*2); // Zero vector
  this->zerosAct = VecD::Zero(hiddenActDim); // Zero vector

  /* For automatic tuning */
  this->prevPerp = REAL_MAX;
  // this->prevModelFileName = this->saveModel(-1);
}

void NMTRNNG::biEncode(const NMTRNNG::Data* data, 
		       NMTRNNG::ThreadArg& arg, 
		       const bool train) { // Encoder for sequence
  int length = data->src.size()-1; // For word

  for (int i = 0; i < arg.srcLen; ++i) {
    if (i == 0) {
      this->enc.forward(this->sourceEmbed.col(data->src[i]), arg.encState[i]);
      this->encRev.forward(this->sourceEmbed.col(data->src[length-i]), arg.encRevState[i]);
    } else {
      this->enc.forward(this->sourceEmbed.col(data->src[i]), arg.encState[i-1], arg.encState[i]);
      this->encRev.forward(this->sourceEmbed.col(data->src[length-i]), arg.encRevState[i-1], arg.encRevState[i]);
    }
    arg.biEncState[i].segment(0, this->hiddenEncDim).noalias() = arg.encState[i]->h;
    arg.biEncState[length-i].segment(this->hiddenEncDim, this->hiddenEncDim).noalias() = arg.encRevState[i]->h;

    if (train) {
      arg.encState[i]->delc = this->zerosEnc; // (!) Initialize here for backward
      arg.encState[i]->delh = this->zerosEnc;
      arg.encRevState[i]->delc = this->zerosEnc;
      arg.encRevState[i]->delh = this->zerosEnc;
    }
  }
  // Affine
  arg.encStateEnd.segment(0, this->hiddenEncDim).noalias() = arg.encState[arg.srcLen-1]->h;
  arg.encStateEnd.segment(this->hiddenEncDim, this->hiddenEncDim).noalias() = arg.encRevState[arg.srcLen-1]->h; 
}

void NMTRNNG::biEncoderBackward2(const NMTRNNG::Data* data,
				 NMTRNNG::ThreadArg& arg,
				 NMTRNNG::Grad& grad) {
  // Backward (this->enc)
  for (int i = arg.srcLen-1; i >= 0; --i){
    this->enc.backward2(arg.encState[i], grad.lstmSrcGrad, this->sourceEmbed.col(data->src[i]), LSTM::WXI);
  }
  for (int i = arg.srcLen-1; i >= 1; --i){
    this->enc.backward2(arg.encState[i], grad.lstmSrcGrad, this->sourceEmbed.col(data->src[i]), LSTM::WXF);
  }
  for (int i = arg.srcLen-1; i >= 0; --i){
    this->enc.backward2(arg.encState[i], grad.lstmSrcGrad, this->sourceEmbed.col(data->src[i]), LSTM::WXO);
  }
  for (int i = arg.srcLen-1; i >= 0; --i){
    this->enc.backward2(arg.encState[i], grad.lstmSrcGrad, this->sourceEmbed.col(data->src[i]), LSTM::WXU);
  }
  for (int i = arg.srcLen-1; i >= 1; --i){
    this->enc.backward2(arg.encState[i], grad.lstmSrcGrad, arg.encState[i-1]->h, LSTM::WHI);
  }
  for (int i = arg.srcLen-1; i >= 1; --i){
    this->enc.backward2(arg.encState[i], grad.lstmSrcGrad, arg.encState[i-1]->h, LSTM::WHF);
  }
  for (int i = arg.srcLen-1; i >= 1; --i){
    this->enc.backward2(arg.encState[i], grad.lstmSrcGrad, arg.encState[i-1]->h, LSTM::WHO);
  }
  for (int i = arg.srcLen-1; i >= 1; --i){
    this->enc.backward2(arg.encState[i], grad.lstmSrcGrad, arg.encState[i-1]->h, LSTM::WHU);
  }
  // Backward (this->encRev)
  for (int i = arg.srcLen-1; i >= 0; --i){
    this->encRev.backward2(arg.encRevState[i], grad.lstmSrcRevGrad, this->sourceEmbed.col(data->src[arg.srcLen-i-1]), LSTM::WXI);
  }
  for (int i = arg.srcLen-1; i >= 1; --i){
    this->encRev.backward2(arg.encRevState[i], grad.lstmSrcRevGrad, this->sourceEmbed.col(data->src[arg.srcLen-i-1]), LSTM::WXF);
  }
  for (int i = arg.srcLen-1; i >= 0; --i){
    this->encRev.backward2(arg.encRevState[i], grad.lstmSrcRevGrad, this->sourceEmbed.col(data->src[arg.srcLen-i-1]), LSTM::WXO);
  }
  for (int i = arg.srcLen-1; i >= 0; --i){
    this->encRev.backward2(arg.encRevState[i], grad.lstmSrcRevGrad, this->sourceEmbed.col(data->src[arg.srcLen-i-1]), LSTM::WXU);
  }
  for (int i = arg.srcLen-1; i >= 1; --i){
    this->encRev.backward2(arg.encRevState[i], grad.lstmSrcRevGrad, arg.encRevState[i-1]->h, LSTM::WHI);
  }
  for (int i = arg.srcLen-1; i >= 1; --i){
    this->encRev.backward2(arg.encRevState[i], grad.lstmSrcRevGrad, arg.encRevState[i-1]->h, LSTM::WHF);
  }
  for (int i = arg.srcLen-1; i >= 1; --i){
    this->encRev.backward2(arg.encRevState[i], grad.lstmSrcRevGrad, arg.encRevState[i-1]->h, LSTM::WHO);
  }
  for (int i = arg.srcLen-1; i >= 1; --i){
    this->encRev.backward2(arg.encRevState[i], grad.lstmSrcRevGrad, arg.encRevState[i-1]->h, LSTM::WHU);
  }
}

void NMTRNNG::decoder(NMTRNNG::ThreadArg& arg, 
		      std::vector<LSTM::State*>& decState,
		      VecD& s_tilde, 
		      const int tgtNum, 
		      const int i, 
		      const bool train) {
  if (i == 0) { // initialize decoder's initial state
    this->decInitAffine.forward(arg.encStateEnd, arg.decState[i]->h);
    arg.decState[i]->c = this->zeros;
  } else { // i >= 1
    // input-feeding approach [Luong et al., EMNLP2015]
    this->dec.forward(this->targetEmbed.col(tgtNum), s_tilde,
		      decState[i-1], decState[i]);
  }
  if (train) {
    arg.decState[i]->delc = this->zeros;
    arg.decState[i]->delh = this->zeros;
    arg.decState[i]->dela = this->zeros;
  }
}

void NMTRNNG::decoderActionBackward2(const NMTRNNG::Data* data,
				     NMTRNNG::ThreadArg& arg,
				     NMTRNNG::Grad& grad) {
  // ActionDecoder (this->act)
  for (int i = arg.actLen-1; i >= 1; --i){
    this->act.backward2(arg.actState[i], grad.lstmActGrad, this->actionEmbed.col(data->action[i-1]), LSTM::WXI);
  }
  for (int i = arg.actLen-1; i >= 1; --i){
    this->act.backward2(arg.actState[i], grad.lstmActGrad, this->actionEmbed.col(data->action[i-1]), LSTM::WXF);
  }
  for (int i = arg.actLen-1; i >= 1; --i){
    this->act.backward2(arg.actState[i], grad.lstmActGrad, this->actionEmbed.col(data->action[i-1]), LSTM::WXO);
  }
  for (int i = arg.actLen-1; i >= 1; --i){
    this->act.backward2(arg.actState[i], grad.lstmActGrad, this->actionEmbed.col(data->action[i-1]), LSTM::WXU);
  }
  for (int i = arg.actLen-1; i >= 1; --i){
    this->act.backward2(arg.actState[i], grad.lstmActGrad, arg.actState[i-1]->h, LSTM::WHI);
  }
  for (int i = arg.actLen-1; i >= 1; --i){
    this->act.backward2(arg.actState[i], grad.lstmActGrad, arg.actState[i-1]->h, LSTM::WHF);
  }
  for (int i = arg.actLen-1; i >= 1; --i){
    this->act.backward2(arg.actState[i], grad.lstmActGrad, arg.actState[i-1]->h, LSTM::WHO);
  }
  for (int i = arg.actLen-1; i >= 1; --i){
    this->act.backward2(arg.actState[i], grad.lstmActGrad, arg.actState[i-1]->h, LSTM::WHU);
  }
}

void NMTRNNG::decoderBackward2(const NMTRNNG::Data* data,
			       NMTRNNG::ThreadArg& arg,
			       NMTRNNG::Grad& grad) {
  for (int i = arg.tgtLen-1; i >= 1; --i){
    this->dec.backward2(arg.decState[i], grad.lstmTgtGrad, this->targetEmbed.col(data->tgt[i-1]), LSTM::WXI);
  }
  for (int i = arg.tgtLen-1; i >= 1; --i){
    this->dec.backward2(arg.decState[i], grad.lstmTgtGrad, this->targetEmbed.col(data->tgt[i-1]), LSTM::WXF);
  }
  for (int i = arg.tgtLen-1; i >= 1; --i){
    this->dec.backward2(arg.decState[i], grad.lstmTgtGrad, this->targetEmbed.col(data->tgt[i-1]), LSTM::WXO);
  }
  for (int i = arg.tgtLen-1; i >= 1; --i){
    this->dec.backward2(arg.decState[i], grad.lstmTgtGrad, this->targetEmbed.col(data->tgt[i-1]), LSTM::WXU);
  }
  for (int i = arg.tgtLen-1; i >= 1; --i){
    this->dec.backward2(arg.decState[i], grad.lstmTgtGrad, arg.decState[i-1]->h, LSTM::WHI);
  }
  for (int i = arg.tgtLen-1; i >= 1; --i){
    this->dec.backward2(arg.decState[i], grad.lstmTgtGrad, arg.decState[i-1]->h, LSTM::WHF);
  }
  for (int i = arg.tgtLen-1; i >= 1; --i){
    this->dec.backward2(arg.decState[i], grad.lstmTgtGrad, arg.decState[i-1]->h, LSTM::WHO);
  }
  for (int i = arg.tgtLen-1; i >= 1; --i){
    this->dec.backward2(arg.decState[i], grad.lstmTgtGrad, arg.decState[i-1]->h, LSTM::WHU);
  }
  for (int i = arg.tgtLen-1; i >= 1; --i){
    this->dec.backward2(arg.decState[i], grad.lstmTgtGrad, arg.s_tilde[i-1], LSTM::WAI);
  }
  for (int i = arg.tgtLen-1; i >= 1; --i){
    this->dec.backward2(arg.decState[i], grad.lstmTgtGrad, arg.s_tilde[i-1], LSTM::WAF);
  }
  for (int i = arg.tgtLen-1; i >= 1; --i){
    this->dec.backward2(arg.decState[i], grad.lstmTgtGrad, arg.s_tilde[i-1], LSTM::WAO);
  }
  for (int i = arg.tgtLen-1; i >= 1; --i){
    this->dec.backward2(arg.decState[i], grad.lstmTgtGrad, arg.s_tilde[i-1], LSTM::WAU);
  }
}

void NMTRNNG::decoderAction(NMTRNNG::ThreadArg& arg, 
			    std::vector<LSTM::State*>& actState,
			    const int actNum, 
			    const int i, 
			    const bool train) {
  if (i == 0) {
    this->actInitAffine.forward(arg.encStateEnd, arg.actState[i]->h);
    arg.actState[i]->c = this->zerosAct;
  } else {
    this->act.forward(this->actionEmbed.col(actNum),
		      actState[i-1], actState[i]); // (xt, prev, cur)
  }
  if (train) {
    arg.actState[i]->delc = this->zerosAct;
    arg.actState[i]->delh = this->zerosAct;
  }
}

void NMTRNNG::decoderReduceLeft(NMTRNNG::Data* data,
				NMTRNNG::ThreadArg& arg,
				const int phraseNum, 
				const int actNum, 
				const int k,
				const bool train) {
  int top;
  int leftNum;
  int rightNum;
  
  this->reduceHeadStack(arg.headStack, top, k);
  this->reduceStack(arg.embedStack, rightNum, leftNum);
  if (train) {
    arg.headList.push_back(top);
    arg.embedList.push_back(leftNum);
    arg.embedList.push_back(rightNum);
  }

  if (rightNum < arg.tgtLen && leftNum < arg.tgtLen) { // word embedding & word embeddding
    this->compositionFunc(arg.embedVec[phraseNum-arg.tgtLen],
			  this->targetEmbed.col(data->tgt[rightNum]), // parent: right
			  this->targetEmbed.col(data->tgt[leftNum]),  // child: left,
			  this->actionEmbed.col(data->action[actNum]),
			  arg.embedVecEnd[phraseNum-arg.tgtLen]);
  } else if (rightNum > (arg.tgtLen-1) && leftNum < arg.tgtLen){
    rightNum -= arg.tgtLen;
    this->compositionFunc(arg.embedVec[phraseNum-arg.tgtLen],
			  arg.embedVec[rightNum],                    // parent: right
			  this->targetEmbed.col(data->tgt[leftNum]), // child: left,
			  this->actionEmbed.col(data->action[actNum]),
			  arg.embedVecEnd[phraseNum-arg.tgtLen]);
  } else if (rightNum < arg.tgtLen && leftNum > (arg.tgtLen-1)){
    leftNum -= arg.tgtLen;
    this->compositionFunc(arg.embedVec[phraseNum-arg.tgtLen],
			  this->targetEmbed.col(data->tgt[rightNum]), // parent: right
			  arg.embedVec[leftNum],                      // child: left,
			  this->actionEmbed.col(data->action[actNum]),
			  arg.embedVecEnd[phraseNum-arg.tgtLen]);
  } else {
    rightNum -= arg.tgtLen;
    leftNum -= arg.tgtLen;
    this->compositionFunc(arg.embedVec[phraseNum-arg.tgtLen],
			  arg.embedVec[rightNum], // parent: right
			  arg.embedVec[leftNum],  // child: left,
			  this->actionEmbed.col(data->action[actNum]),
			  arg.embedVecEnd[phraseNum-arg.tgtLen]);
  }

  this->outBuf.forward(arg.embedVec[phraseNum-arg.tgtLen],
		       arg.outBufState[top], arg.outBufState[k]); // (xt, prev, cur)
  if (train) {
    arg.outBufState[k]->delc = this->zeros;
    arg.outBufState[k]->delh = this->zeros;
  }
  arg.embedStack.push(phraseNum);
}

void NMTRNNG::decoderReduceRight(NMTRNNG::Data* data,
				 NMTRNNG::ThreadArg& arg,
				 const int phraseNum, 
				 const int actNum, 
				 const int k, 
				 const bool train) {
  int top;
  int leftNum;
  int rightNum;
  
  this->reduceHeadStack(arg.headStack, top, k);
  this->reduceStack(arg.embedStack, rightNum, leftNum);
  if (train) {
    arg.headList.push_back(top);
    arg.embedList.push_back(leftNum);
    arg.embedList.push_back(rightNum);
  }

  if (rightNum < arg.tgtLen && leftNum < arg.tgtLen) { // word Embed & word Embed
    this->compositionFunc(arg.embedVec[phraseNum-arg.tgtLen],
			  this->targetEmbed.col(data->tgt[leftNum]),  // parent: left
			  this->targetEmbed.col(data->tgt[rightNum]), // child: right,
			  this->actionEmbed.col(data->action[actNum]),
			  arg.embedVecEnd[phraseNum-arg.tgtLen]);
  } else if (rightNum > (arg.tgtLen-1) && leftNum < arg.tgtLen){
    rightNum -= arg.tgtLen;
    this->compositionFunc(arg.embedVec[phraseNum-arg.tgtLen],
			  this->targetEmbed.col(data->tgt[leftNum]), // parent: left,
			  arg.embedVec[rightNum],                    // child: right
			  this->actionEmbed.col(data->action[actNum]),
			  arg.embedVecEnd[phraseNum-arg.tgtLen]);
  } else if (rightNum < arg.tgtLen && leftNum > (arg.tgtLen-1)){
    leftNum -= arg.tgtLen;
    this->compositionFunc(arg.embedVec[phraseNum-arg.tgtLen],
			  arg.embedVec[leftNum],                      // parent: left,
			  this->targetEmbed.col(data->tgt[rightNum]), // child: right
			  this->actionEmbed.col(data->action[actNum]),
			  arg.embedVecEnd[phraseNum-arg.tgtLen]);
  } else {
    rightNum -= arg.tgtLen;
    leftNum -= arg.tgtLen;
    this->compositionFunc(arg.embedVec[phraseNum-arg.tgtLen],
			  arg.embedVec[leftNum],  // parent: left,
			  arg.embedVec[rightNum], // child: right
			  this->actionEmbed.col(data->action[actNum]),
			  arg.embedVecEnd[phraseNum-arg.tgtLen]);
  }

  this->outBuf.forward(arg.embedVec[phraseNum-arg.tgtLen],
		       arg.outBufState[top], arg.outBufState[k]); // (xt, prev, cur)
  if (train) {
    arg.outBufState[k]->delc = this->zeros;
    arg.outBufState[k]->delh = this->zeros;
  }
  arg.embedStack.push(phraseNum);
}

void NMTRNNG::compositionFunc(VecD& c, 
			      const VecD& head, 
			      const VecD& dependent, 
			      const VecD& relation, 
			      VecD& embedVecEnd) {
  embedVecEnd.segment(0, this->inputDim).noalias() = head;
  embedVecEnd.segment(this->inputDim, this->inputDim).noalias() = dependent; 
  embedVecEnd.segment(this->inputDim*2, this->inputActDim).noalias() = relation; 
  this->embedVecAffine.forward(embedVecEnd, c);
}

void NMTRNNG::reduceHeadStack(std::stack<int>& stack,
			      int& top, 
			      const int k) {
  stack.pop();
  stack.pop();
  top = stack.top();
  stack.push(k);
}

void NMTRNNG::reduceStack(std::stack<int>& stack,
			  int& right, 
			  int& left) {
  right = stack.top();
  stack.pop();
  left = stack.top();
  stack.pop();
}

void NMTRNNG::decoderAttention(NMTRNNG::ThreadArg& arg,
			       const LSTM::State* decState,
			       VecD& contextSeq, 
			       VecD& s_tilde, 
			       VecD& stildeEnd) { // CalcLoss / Test
  /* Attention */
  // sequence
  contextSeq = this->zeros2;

  this->calculateAlpha(arg, decState);

  for (int j = 0; j < arg.srcLen; ++j) {
    contextSeq.noalias() += arg.alphaSeqVec.coeff(j, 0) * arg.biEncState[j];
  }

  stildeEnd.segment(0, this->hiddenDim).noalias() = decState->h;
  stildeEnd.segment(this->hiddenDim, this->hiddenEncDim*2).noalias() = contextSeq; 

  this->stildeAffine.forward(stildeEnd, s_tilde);
}

void NMTRNNG::decoderAttention(NMTRNNG::ThreadArg& arg, 
			       const int i,
			       const bool train) { // Train
  /* Attention */
  // sequence
  arg.contextSeqList[i] = this->zeros2;

  this->calculateAlpha(arg, arg.decState[i], i);

  for (int j = 0; j < arg.srcLen; ++j) {
    arg.contextSeqList[i].noalias() += arg.alphaSeq.coeff(j, i) * arg.biEncState[j];
  }
  arg.stildeEnd[i].segment(0, this->hiddenDim).noalias() = arg.decState[i]->h;
  arg.stildeEnd[i].segment(this->hiddenDim, this->hiddenEncDim*2).noalias() = arg.contextSeqList[i];

  this->stildeAffine.forward(arg.stildeEnd[i], arg.s_tilde[i]);
}

struct sort_pred {
  bool operator()(const NMTRNNG::DecCandidate left, const NMTRNNG::DecCandidate right) {
    return left.score > right.score;
  }
};

struct sort_predNMTRNNG {
  bool operator()(const NMTRNNG::DecCandidate left, const NMTRNNG::DecCandidate right) {
    return (left.score + left.scoreAct) > (right.score + right.scoreAct);
  }
};

void NMTRNNG::translate(NMTRNNG::Data* data, 
			NMTRNNG::ThreadArg& arg, 
			std::vector<int>& translation, 
			const bool train) {
  const Real minScore = -1.0e+05;
  const int maxLength = this->maxLen;
  const int beamSize = arg.candidate.size();
  int showNum;

  if ((int)arg.candidate.size() > 1) {
    showNum = 5; 
  } else {
    showNum = 1;
  }
  MatD score = MatD(this->targetEmbed.cols(), beamSize);
  std::vector<NMTRNNG::DecCandidate> candidateTmp(beamSize);

  for (auto it = arg.candidate.begin(); it != arg.candidate.end(); ++it) {
    it->init(*this);
  }
  arg.init(*this, data, false);
  this->biEncode(data, arg, false); // encoder

  for (int i = 0; i < maxLength; ++i) {
    for (int j = 0; j < beamSize; ++j) {
      VecD stildeEnd = VecD(this->hiddenDim + this->hiddenEncDim*2);
      VecD encStateEnd = VecD(this->hiddenEncDim*2);
      if (arg.candidate[j].stop) {
	score.col(j).fill(arg.candidate[j].score);
	continue;
      }
      if (i == 0) {
	VecD decInitStateEnd = VecD(this->hiddenEncDim*2);
	decInitStateEnd.segment(0, this->hiddenEncDim).noalias() = arg.encState[arg.srcLen-1]->h;
	decInitStateEnd.segment(this->hiddenEncDim, this->hiddenEncDim).noalias() = arg.encRevState[arg.srcLen-1]->h; 
	this->decInitAffine.forward(decInitStateEnd, arg.candidate[j].curDec.h);
	arg.candidate[j].curDec.c = this->zeros;
      } else {	
	arg.candidate[j].prevDec.h = arg.candidate[j].curDec.h;
	arg.candidate[j].prevDec.c = arg.candidate[j].curDec.c;
	this->dec.forward(this->targetEmbed.col(arg.candidate[j].generatedTarget[i-1]), arg.candidate[j].s_tilde,
			  &arg.candidate[j].prevDec, &arg.candidate[j].curDec);
      }
      this->decoderAttention(arg, &arg.candidate[j].curDec, arg.candidate[j].contextSeq, arg.candidate[j].s_tilde, stildeEnd);
      if (!this->useBlackOut) {
	this->softmax.calcDist(arg.candidate[j].s_tilde, arg.candidate[j].targetDist);
      } else {
	this->blackOut.calcDist(arg.candidate[j].s_tilde, arg.candidate[j].targetDist);
      }
      score.col(j).array() = arg.candidate[j].score + arg.candidate[j].targetDist.array().log();
    }
    for (int j = 0, row, col; j < beamSize; ++j) {
      score.maxCoeff(&row, &col); // Greedy
      candidateTmp[j] = arg.candidate[col];
      candidateTmp[j].score = score.coeff(row, col);

      if (candidateTmp[j].stop) { // if "EOS" comes up...
	score.col(col).fill(minScore);
	continue;
      }

      candidateTmp[j].generatedTarget.push_back(row);

      if (row == this->targetVoc.eosIndex) {
	candidateTmp[j].stop = true;
      }

      if (i == 0) {
	score.row(row).fill(minScore);
      } else {
	score.coeffRef(row, col) = minScore;
      }
    }

    arg.candidate = candidateTmp;

    std::sort(arg.candidate.begin(), arg.candidate.end(), sort_pred());

    if (arg.candidate[0].generatedTarget.back() == this->targetVoc.eosIndex) {
      break;
    }
  }

  if (train) {
    for (auto it = data->src.begin(); it != data->src.end(); ++it) {
      std::cout << this->sourceVoc.tokenList[*it]->str << " "; 
    }
    std::cout << std::endl;

    for (int i = 0; i < showNum; ++i) {
      std::cout << i+1 << " (" << arg.candidate[i].score << "): ";

      for (auto it = arg.candidate[i].generatedTarget.begin(); it != arg.candidate[i].generatedTarget.end(); ++it) {
	std::cout << this->targetVoc.tokenList[*it]->str << " ";
      }
      std::cout << std::endl;
    }

    for (auto it = data->src.begin(); it != data->src.end(); ++it) {
      std::cout << this->sourceVoc.tokenList[*it]->str << " ";
    }
    std::cout << std::endl;
  } else {
    this->makeTrans(arg.candidate[0].generatedTarget, data->trans);
  }
}

void NMTRNNG::translateWithAction(NMTRNNG::Data* data, 
				  NMTRNNG::ThreadArg& arg, 
				  std::vector<int>& translation, 
				  const int beamSizeA, 
				  const bool train) {
  const Real minScore = -1.0e+05;
  const int maxLength = this->maxLen;
  const int beamSize = arg.candidate.size();

  MatD score = MatD(this->targetEmbed.cols(), beamSize);
  MatD scoreAct = MatD(this->actionEmbed.cols(), beamSize);
  std::vector<NMTRNNG::DecCandidate> candidateTmp(beamSize);

  for (auto it = arg.candidate.begin(); it != arg.candidate.end(); ++it) {
    it->init(*this);
  }
  arg.init(*this, data, false);
  this->biEncode(data, arg, false); // encoder

  VecD decInitStateEnd = VecD(this->hiddenEncDim*2);
  VecD actInit = VecD(this->hiddenActDim);
  VecD outBufInit = VecD(this->hiddenDim);

  /* Decoder part starts */
  decInitStateEnd.segment(0, this->hiddenEncDim).noalias() = arg.encState[arg.srcLen-1]->h;
  decInitStateEnd.segment(this->hiddenEncDim, this->hiddenEncDim).noalias() = arg.encRevState[arg.srcLen-1]->h;
  // Initialize Decoder; j == 0
  this->decInitAffine.forward(decInitStateEnd, arg.candidate[0].curDec.h);
  arg.candidate[0].curDec.c = this->zeros;
  arg.candidate[0].utEnd.segment(0, this->hiddenDim).noalias() = arg.candidate[0].curDec.h;

  // Initialize Out Buffer; k == 0
  this->outBufInitAffine.forward(decInitStateEnd, arg.candidate[0].outBufState[arg.candidate[0].k]->h);
  arg.candidate[0].outBufState[arg.candidate[0].k]->c = this->zeros;
  arg.candidate[0].headStack.push(arg.candidate[0].k);
  arg.candidate[0].utEnd.segment(this->hiddenDim, this->hiddenDim).noalias() = arg.candidate[0].outBufState[arg.candidate[0].k]->h;
  ++arg.candidate[0].k;

  // Initilize Action: i == 0
  this->actInitAffine.forward(decInitStateEnd, arg.candidate[0].curAct.h);
  arg.candidate[0].curAct.c = this->zerosAct;
  arg.candidate[0].utEnd.segment(this->hiddenDim*2, this->hiddenActDim).noalias() = arg.candidate[0].curAct.h;
  // forward
  this->utAffine.forward(arg.candidate[0].utEnd, arg.candidate[0].ut);

  // Softmax & Score
  this->softmaxAct.calcDist(arg.candidate[0].ut, arg.candidate[0].targetActDist);
  scoreAct.col(0).array() = arg.candidate[0].scoreAct + arg.candidate[0].targetActDist.array().log();

  int row, col;
  scoreAct.col(0).maxCoeff(&row, &col);
  arg.candidate[0].scoreAct = scoreAct.coeff(row, col); // (row, col) = (0, 0)

  arg.candidate[0].generatedAction.push_back(row); // 0: SHIFT

  for (int i = 1; i < beamSize; ++i) {
    arg.candidate[i] = arg.candidate[0];
  }

  for (int i = 0; i < maxLength; ++i) {
    for (int j = 0; j < beamSize; ++j) {
      VecD stildeEnd = VecD(this->hiddenDim + this->hiddenEncDim*2);

      if (arg.candidate[j].stop) {
	score.col(j).fill(arg.candidate[j].score);
	  continue;
	}
      if (i == 0) {
      } else {
	arg.candidate[j].prevDec.h = arg.candidate[j].curDec.h;
	arg.candidate[j].prevDec.c = arg.candidate[j].curDec.c;
	this->dec.forward(this->targetEmbed.col(arg.candidate[j].generatedTarget[i-1]), arg.candidate[j].s_tilde,
			  &arg.candidate[j].prevDec, &arg.candidate[j].curDec);
      }
      this->decoderAttention(arg, &arg.candidate[j].curDec, arg.candidate[j].contextSeq, arg.candidate[j].s_tilde, stildeEnd);
      if (!this->useBlackOut) {
	this->softmax.calcDist(arg.candidate[j].s_tilde, arg.candidate[j].targetDist);
      } else {
	this->blackOut.calcDist(arg.candidate[j].s_tilde, arg.candidate[j].targetDist);
      }
      score.col(j).array() = arg.candidate[j].score + arg.candidate[j].targetDist.array().log();
    }
    for (int j = 0, row, col; j < beamSize; ++j) {
      score.maxCoeff(&row, &col); // Greedy
      candidateTmp[j] = arg.candidate[col];
      candidateTmp[j].score = score.coeff(row, col);

      if (candidateTmp[j].stop) { // if "EOS" comes up...
	score.col(col).fill(minScore);
	continue;
      }

      candidateTmp[j].generatedTarget.push_back(row);
      // std::cout << this->targetVoc.tokenList[row]->str << " ";
      // utEnd
      candidateTmp[j].utEnd.segment(0, this->hiddenDim).noalias() = candidateTmp[j].curDec.h;
      candidateTmp[j].headStack.push(candidateTmp[j].k);
      this->outBuf.forward(this->targetEmbed.col(row), 
			   candidateTmp[j].outBufState[candidateTmp[j].k-1], 
			   candidateTmp[j].outBufState[candidateTmp[j].k]);
      candidateTmp[j].embedStack.push(i);

      // SoftmaxAct calculation (o: output buffer, s: stack, and h: action)
      candidateTmp[j].utEnd.segment(this->hiddenDim, this->hiddenDim).noalias() = candidateTmp[j].outBufState[candidateTmp[j].k]->h;
      ++candidateTmp[j].k;


      if (row == this->targetVoc.eosIndex) {
	candidateTmp[j].stop = true;
      }

      if (i == 0) {
	score.row(row).fill(minScore);
      } else {
	score.coeffRef(row, col) = minScore;
      }
    }

    arg.candidate = candidateTmp;

    std::sort(arg.candidate.begin(), arg.candidate.end(), sort_pred());

    // Action
    for (int j = 0, row, col; j < beamSize; j++) {
      bool shift = false;
      while (!shift) {
	if (arg.candidate[j].k >= 300) {
	  return;
	}

	arg.candidate[j].prevAct.h = arg.candidate[j].curAct.h;
	arg.candidate[j].prevAct.c = arg.candidate[j].curAct.c;
	this->act.forward(this->actionEmbed.col(arg.candidate[j].generatedAction.back()),
			  &arg.candidate[j].prevAct, &arg.candidate[j].curAct); // (xt, prev, cur)
	arg.candidate[j].utEnd.segment(this->hiddenDim*2, this->hiddenActDim).noalias() = arg.candidate[j].curAct.h; 
	// forward
	this->utAffine.forward(arg.candidate[j].utEnd, arg.candidate[j].ut);

	// Softmax & Score
	this->softmaxAct.calcDist(arg.candidate[j].ut, arg.candidate[j].targetActDist);
	scoreAct.col(j).array() = arg.candidate[j].scoreAct + arg.candidate[j].targetActDist.array().log();
	if (arg.candidate[j].headStack.size() > 2) {
	  scoreAct.col(j).maxCoeff(&row, &col);
	} else {
	  // scoreAct.row(0).maxCoeff(&row, &col);
	  row = 0;
	  col = j;
	}

	arg.candidate[j].scoreAct = scoreAct.coeff(row, col); // (row, col) = (row, j)
	arg.candidate[j].generatedAction.push_back(row);
	// std::cout << this->actionVoc.tokenList[row]->str << " ";
	
	if (arg.candidate[j].generatedTarget.back() == this->targetVoc.eosIndex) {
	  shift = true;
	} 
	if (this->actionVoc.tokenList[row]->action == 0) { // shift
	  shift = true;
	} else if (this->actionVoc.tokenList[row]->action == 1) { // 1: Reduce-Left
	  int top;
	  int leftNum;
	  int rightNum;
	  this->reduceHeadStack(arg.candidate[j].headStack, top, arg.candidate[j].k);
	  this->reduceStack(arg.candidate[j].embedStack, rightNum, leftNum);

	  if (rightNum < arg.candidate[j].tgtLen && leftNum < arg.candidate[j].tgtLen) { // word embedding & word embeddding
	    this->compositionFunc(arg.candidate[j].embedVec[arg.candidate[j].phraseNum-arg.candidate[j].tgtLen],
				  this->targetEmbed.col(arg.candidate[j].generatedTarget[rightNum]), // parent: right
				  this->targetEmbed.col(arg.candidate[j].generatedTarget[leftNum]),  // child: left,
				  this->actionEmbed.col(arg.candidate[j].generatedAction.back()),
				  arg.candidate[j].embedVecEnd[arg.candidate[j].phraseNum-arg.candidate[j].tgtLen]);
	  } else if (rightNum > (arg.candidate[j].tgtLen-1) && leftNum < arg.candidate[j].tgtLen){
	    rightNum -= arg.candidate[j].tgtLen;
	    this->compositionFunc(arg.candidate[j].embedVec[arg.candidate[j].phraseNum-arg.candidate[j].tgtLen],
				  arg.candidate[j].embedVec[rightNum],                              // parent: right
				  this->targetEmbed.col(arg.candidate[j].generatedTarget[leftNum]), // child: left,
				  this->actionEmbed.col(arg.candidate[j].generatedAction.back()),
				  arg.candidate[j].embedVecEnd[arg.candidate[j].phraseNum-arg.candidate[j].tgtLen]);
	  } else if (rightNum < arg.candidate[j].tgtLen && leftNum > (arg.candidate[j].tgtLen-1)){
	    leftNum -= arg.candidate[j].tgtLen;
	    this->compositionFunc(arg.candidate[j].embedVec[arg.candidate[j].phraseNum-arg.candidate[j].tgtLen],
				  this->targetEmbed.col(arg.candidate[j].generatedTarget[rightNum]), // parent: right
				  arg.candidate[j].embedVec[leftNum],                                // child: left,
				  this->actionEmbed.col(arg.candidate[j].generatedAction.back()),
				  arg.candidate[j].embedVecEnd[arg.candidate[j].phraseNum-arg.candidate[j].tgtLen]);
	  } else {
	    rightNum -= arg.candidate[j].tgtLen;
	    leftNum -= arg.candidate[j].tgtLen;
	    this->compositionFunc(arg.candidate[j].embedVec[arg.candidate[j].phraseNum-arg.candidate[j].tgtLen],
				  arg.candidate[j].embedVec[rightNum], // parent: right
				  arg.candidate[j].embedVec[leftNum],  // child: left,
				  this->actionEmbed.col(arg.candidate[j].generatedAction.back()),
				  arg.candidate[j].embedVecEnd[arg.candidate[j].phraseNum-arg.candidate[j].tgtLen]);
	  }

	  this->outBuf.forward(arg.candidate[j].embedVec[arg.candidate[j].phraseNum-arg.candidate[j].tgtLen],
			       arg.candidate[j].outBufState[top], arg.candidate[j].outBufState[arg.candidate[j].k]); // (xt, prev, cur)
	  arg.candidate[j].embedStack.push(arg.candidate[j].phraseNum);
	  ++arg.candidate[j].phraseNum;
	  arg.candidate[j].utEnd.segment(this->hiddenDim, this->hiddenDim).noalias() = arg.candidate[j].outBufState[arg.candidate[j].k]->h;
	  ++arg.candidate[j].k;

	} else if (this->actionVoc.tokenList[row]->action == 2) { // 2: Reduce-Right
	  int top;
	  int leftNum;
	  int rightNum;
	  this->reduceHeadStack(arg.candidate[j].headStack, top, arg.candidate[j].k);
	  this->reduceStack(arg.candidate[j].embedStack, rightNum, leftNum);

	  if (rightNum < arg.candidate[j].tgtLen && leftNum < arg.candidate[j].tgtLen) { // word Embed & word Embed
	    this->compositionFunc(arg.candidate[j].embedVec[arg.candidate[j].phraseNum-arg.candidate[j].tgtLen],
				  this->targetEmbed.col(arg.candidate[j].generatedTarget[leftNum]),  // parent: left
				  this->targetEmbed.col(arg.candidate[j].generatedTarget[rightNum]), // child: right,
				  this->actionEmbed.col(arg.candidate[j].generatedAction.back()),
				  arg.candidate[j].embedVecEnd[arg.candidate[j].phraseNum-arg.candidate[j].tgtLen]);
	  } else if (rightNum > (arg.candidate[j].tgtLen-1) && leftNum < arg.candidate[j].tgtLen){
	    rightNum -= arg.candidate[j].tgtLen;
	    this->compositionFunc(arg.candidate[j].embedVec[arg.candidate[j].phraseNum-arg.candidate[j].tgtLen],
				  this->targetEmbed.col(arg.candidate[j].generatedTarget[leftNum]), // parent: left,
				  arg.candidate[j].embedVec[rightNum],                              // child: right
				  this->actionEmbed.col(arg.candidate[j].generatedAction.back()),
				  arg.candidate[j].embedVecEnd[arg.candidate[j].phraseNum-arg.candidate[j].tgtLen]);
	  } else if (rightNum < arg.candidate[j].tgtLen && leftNum > (arg.candidate[j].tgtLen-1)){
	    leftNum -= arg.candidate[j].tgtLen;
	    this->compositionFunc(arg.candidate[j].embedVec[arg.candidate[j].phraseNum-arg.candidate[j].tgtLen],
				  arg.candidate[j].embedVec[leftNum],                                // parent: left,
				  this->targetEmbed.col(arg.candidate[j].generatedTarget[rightNum]), // child: right
				  this->actionEmbed.col(arg.candidate[j].generatedAction.back()),
				  arg.candidate[j].embedVecEnd[arg.candidate[j].phraseNum-arg.candidate[j].tgtLen]);
	  } else {
	    rightNum -= arg.candidate[j].tgtLen;
	    leftNum -= arg.candidate[j].tgtLen;
	    this->compositionFunc(arg.candidate[j].embedVec[arg.candidate[j].phraseNum-arg.candidate[j].tgtLen],
				  arg.candidate[j].embedVec[leftNum],  // parent: left,
				  arg.candidate[j].embedVec[rightNum], // child: right
				  this->actionEmbed.col(arg.candidate[j].generatedAction.back()),
				  arg.candidate[j].embedVecEnd[arg.candidate[j].phraseNum-arg.candidate[j].tgtLen]);
	  }
	  this->outBuf.forward(arg.candidate[j].embedVec[arg.candidate[j].phraseNum-arg.candidate[j].tgtLen],
			       arg.candidate[j].outBufState[top], arg.candidate[j].outBufState[arg.candidate[j].k]); // (xt, prev, cur)
	  arg.candidate[j].embedStack.push(arg.candidate[j].phraseNum);
	  ++arg.candidate[j].phraseNum;
	  arg.candidate[j].utEnd.segment(this->hiddenDim, this->hiddenDim).noalias() = arg.candidate[j].outBufState[arg.candidate[j].k]->h;
	  ++arg.candidate[j].k;
	}  else {
	  print("Error Non-Shift/Reduce");
	  exit(2);
	}
      }

      if (arg.candidate[0].generatedTarget.back() == this->targetVoc.eosIndex) {
	break;
      }
    }

    if (arg.candidate[0].generatedTarget.back() == this->targetVoc.eosIndex) {
      break;
    } 
  }
  std::cout << std::endl;

  if (train) {
  } else {
    for (auto it = data->src.begin(); it != data->src.end(); ++it) {
      std::cout << this->sourceVoc.tokenList[*it]->str << " "; 
    }
    std::cout << std::endl;

    for (auto it = data->tgt.begin(); it != data->tgt.end(); ++it) {
      std::cout << this->targetVoc.tokenList[*it]->str << " "; 
    }
    std::cout << std::endl;

    for (auto it = arg.candidate[0].generatedTarget.begin(); it != arg.candidate[0].generatedTarget.end(); ++it) {
      std::cout << this->targetVoc.tokenList[*it]->str << " "; 
    }
    std::cout << std::endl;

    int i = 0;
    for (auto it = arg.candidate[0].generatedTarget.begin(); it != arg.candidate[0].generatedTarget.end(); ++it) {
      std::cout << this->actionVoc.tokenList[arg.candidate[0].generatedAction[i]]->str << " ";
      ++i;
      std::cout << this->targetVoc.tokenList[*it]->str << " ";
      while (true) {
	if (arg.candidate[0].generatedAction[i] == 0) { // SHIFT
	  break;
	}
	std::cout << this->actionVoc.tokenList[arg.candidate[0].generatedAction[i]]->str << " ";
	++i;
	if (i > (int)arg.candidate[0].generatedAction.size()-1) {
	  break;
	}
      }
    }
    std::cout << std::endl;
  }
  exit(1);
}

void NMTRNNG::train(NMTRNNG::Data* data,
		    NMTRNNG::ThreadArg& arg, 
		    NMTRNNG::Grad& grad, 
		    const bool train = true) {
  int length = data->src.size()-1; // source words
  int top = 0;
  int j = 0;
  int k = 0;
  int phraseNum = data->tgt.size(); // mapping a phrase
  int leftNum = -1;
  int rightNum = -1;
  int tgtRightNum = -1;
  int tgtLeftNum = -1;
  int actNum = -1;

  arg.init(*this, data, train);
  this->biEncode(data, arg, train); // encoder

  // Out Buffer (=> Stack); k == 0
  this->outBufInitAffine.forward(arg.encStateEnd, arg.outBufState[k]->h);
  arg.outBufState[k]->c = this->zeros;

  if (train) {
    arg.outBufState[k]->delc = this->zeros;
    arg.outBufState[k]->delh = this->zeros;
  }
  arg.headStack.push(k);
  ++k;

  for (int i = 0; i < arg.actLen; ++i, ++k) { // SHIFT-REDUCE
    // SoftmaxAct calculation
    actNum = data->action[i];
    this->decoderAction(arg, arg.actState, data->action[i-1], i, train); // PUSH

    if (this->actionVoc.tokenList[actNum]->action == 0) { // 0: Shift
      arg.headStack.push(k);
      // 1) Let NMT's decoder proceed one step; PUSH
      this->decoder(arg, arg.decState, arg.s_tilde[j-1], data->tgt[j-1], j, train);
      /* Attention */
      this->decoderAttention(arg, j, train);

      // 2) Let the output buffer proceed one step, though the computed unit is not used at this step; PUSH
      this->outBuf.forward(this->targetEmbed.col(data->tgt[j]), 
			   arg.outBufState[k-1], arg.outBufState[k]);
      if (train) { 
	arg.outBufState[k]->delc = this->zeros;
	arg.outBufState[k]->delh = this->zeros;
      }
      arg.embedStack.push(j);

      // SoftmaxAct calculation (o: output buffer, s: stack, and h: action)
      arg.utEnd[i].segment(0, this->hiddenDim).noalias() = arg.decState[j]->h;

      ++j;

    } else if (this->actionVoc.tokenList[actNum]->action == 1) { // 1: Reduce-Left
      this->decoderReduceLeft(data, arg, phraseNum, i-1, k, true);
      ++phraseNum;

      // SoftmaxAct calculation (o: output buffer, s: stack, and h: action)
      arg.utEnd[i].segment(0, this->hiddenDim).noalias() = arg.decState[j-1]->h;

    } else if (this->actionVoc.tokenList[actNum]->action == 2) { // 2: Reduce-Right
      this->decoderReduceRight(data, arg, phraseNum, i-1, k, true);
      ++phraseNum;

      // SoftmaxAct calculation (o: output buffer, s: stack, and h: action)
      arg.utEnd[i].segment(0, this->hiddenDim).noalias() = arg.decState[j-1]->h;

    } else {
      print("Error Non-Shift/Reduce");
      exit(2);
    }

    arg.utEnd[i].segment(this->hiddenDim, this->hiddenDim).noalias() = arg.outBufState[k-1]->h; 
    arg.utEnd[i].segment(this->hiddenDim*2, this->hiddenActDim).noalias() = arg.actState[i]->h; 

    this->utAffine.forward(arg.utEnd[i], arg.ut[i]);
  }

  // Backward (Action)
  for (int i = 0; i < arg.actLen; ++i) {
    this->softmaxAct.calcDist(arg.ut[i], arg.actionDist);
    arg.loss += this->softmaxAct.calcLoss(arg.actionDist, data->action[i]);
    this->softmaxAct.backward(arg.ut[i], arg.actionDist, data->action[i], arg.del_ut[i], grad.softmaxActGrad);
  }

  // Backward (Decoder)
  if (!this->useBlackOut) {
    for (int i = 0; i < arg.tgtLen; ++i) {
      this->softmax.calcDist(arg.s_tilde[i], arg.targetDist);
      arg.loss += this->softmax.calcLoss(arg.targetDist, data->tgt[i]);
      this->softmax.backward(arg.s_tilde[i], arg.targetDist, data->tgt[i], arg.del_stilde[i], grad.softmaxGrad);
    }
  } else { // BlackOut with negative samples shared
    this->blackOut.sampling2(arg.blackOutState[0], this->targetVoc.unkIndex); // unk
    for (int i = 0; i < arg.tgtLen; ++i){
      // word prediction
      arg.blackOutState[0].sample[0] = data->tgt[i];
      arg.blackOutState[0].weight.col(0) = this->blackOut.weight.col(data->tgt[i]);
      arg.blackOutState[0].bias.coeffRef(0, 0) = this->blackOut.bias.coeff(data->tgt[i], 0);
      this->blackOut.calcSampledDist2(arg.s_tilde[i], arg.targetDistVec[i], arg.blackOutState[0]);
      arg.loss += this->blackOut.calcSampledLoss(arg.targetDistVec[i]);
    }
    this->blackOut.backward_1(arg.tgtLen, data->tgt, arg.targetDistVec, arg.blackOutState, arg.del_stilde);
    this->blackOut.backward_2(arg.tgtLen, data->tgt, arg.s_tilde, arg.blackOutState, grad.blackOutGrad);
  }

  /* -- Backpropagation starts -- */
  --j;
  --k;
  --phraseNum;
  for (int i = arg.actLen-1; i >= 0; --i, --k) {
    this->utAffine.backward1(arg.ut[i], arg.del_ut[i], arg.del_ut[i], arg.del_utEnd, grad.utAffineGrad);
    arg.outBufState[k-1]->delh.noalias() += arg.del_utEnd.segment(this->hiddenDim, this->hiddenDim);
    arg.actState[i]->delh.noalias() += arg.del_utEnd.segment(this->hiddenDim*2, this->hiddenActDim);

    actNum = data->action[i];
    if (this->actionVoc.tokenList[actNum]->action == 0) { // 0: Shift
      arg.decState[j]->delh.noalias() += arg.del_utEnd.segment(0, this->hiddenDim);

      // 2) Output Buffer
      this->outBuf.backward(arg.outBufState[k-1], arg.outBufState[k], 
			    grad.lstmOutBufGrad, this->targetEmbed.col(data->tgt[j]));
      if (grad.targetEmbed.count(data->tgt[j])) {
	grad.targetEmbed.at(data->tgt[j]).noalias() += arg.outBufState[k]->delx;
      } else {
	grad.targetEmbed[data->tgt[j]].noalias() = arg.outBufState[k]->delx;
      }

      // 1) NMT's Decoder
      /* Attention's Backpropagation */
      // arg.del_stilde
      if (j < (arg.tgtLen-1)) {
	arg.del_stilde[j].noalias() += arg.decState[j+1]->dela; // add gradients to the previous arg.del_stilde 
	                                                        // by input-feeding [Luong et al., EMNLP2015]
      } else {}

      this->stildeAffine.backward1(arg.s_tilde[j], arg.del_stilde[j], arg.del_stilde[j], arg.del_stildeEnd, grad.stildeAffineGrad);
      arg.decState[j]->delh.noalias() += arg.del_stildeEnd.segment(0, this->hiddenDim);

      // del_contextSeq
      for (int j2 = 0; j2 < arg.srcLen; ++j2) { // Seq
	arg.del_alphaSeqTmp = arg.alphaSeq.coeff(j2, j) * arg.del_stildeEnd.segment(this->hiddenDim, this->hiddenEncDim*2);
	arg.encState[j2]->delh.noalias() += arg.del_alphaSeqTmp.segment(0, this->hiddenEncDim);
	arg.encRevState[arg.srcLen-j2-1]->delh.noalias() += arg.del_alphaSeqTmp.segment(this->hiddenEncDim, this->hiddenEncDim);
	arg.del_alphaSeq.coeffRef(j2, 0) = arg.del_stildeEnd.segment(this->hiddenDim, this->hiddenEncDim*2).dot(arg.biEncState[j2]);
      } 
      arg.del_alignScore = arg.alphaSeq.col(j).array()*(arg.del_alphaSeq.array()-arg.alphaSeq.col(j).dot(arg.del_alphaSeq)); // X.array() - scalar; np.array() -= 1

      for (int j2 = 0; j2 < arg.srcLen; ++j2) {
	arg.del_WgeneralTmp = this->Wgeneral.transpose()*arg.decState[j]->h;
	arg.encState[j2]->delh.noalias() += arg.del_WgeneralTmp.segment(0, this->hiddenEncDim) * arg.del_alignScore.coeff(j2, 0);
	arg.encRevState[arg.srcLen-j2-1]->delh.noalias() += arg.del_WgeneralTmp.segment(this->hiddenEncDim, this->hiddenEncDim) * arg.del_alignScore.coeff(j2, 0);

	arg.decState[j]->delh.noalias() += (this->Wgeneral*arg.biEncState[j2])*arg.del_alignScore.coeff(j2, 0);
	grad.Wgeneral += arg.del_alignScore.coeff(j2, 0)*arg.decState[j]->h*arg.biEncState[j2].transpose();
      }

      if (j > 0) {
	// Backward
	this->dec.backward1(arg.decState[j-1], arg.decState[j], 
			    grad.lstmTgtGrad, this->targetEmbed.col(data->tgt[j-1]), arg.s_tilde[j-1]);
	if (grad.targetEmbed.count(data->tgt[j-1])) {
	  grad.targetEmbed.at(data->tgt[j-1]).noalias() += arg.decState[j]->delx;
	} else {
	  grad.targetEmbed[data->tgt[j-1]].noalias() = arg.decState[j]->delx;
	}
      } else {}
      
      --j;

    } else if (this->actionVoc.tokenList[actNum]->action == 1) { // 1: Reduce-Left
      arg.decState[j]->delh.noalias() += arg.del_utEnd.segment(0, this->hiddenDim);

      top = arg.headList.back();
      arg.headList.pop_back();

      rightNum = arg.embedList.back();
      arg.embedList.pop_back();
      leftNum = arg.embedList.back();
      arg.embedList.pop_back();

      this->outBuf.backward(arg.outBufState[top], arg.outBufState[k], 
			    grad.lstmOutBufGrad, arg.embedVec[phraseNum-arg.tgtLen]);

      if (arg.del_embedVec.count(phraseNum-arg.tgtLen)) {
	arg.del_embedVec.at(phraseNum-arg.tgtLen).noalias() += arg.outBufState[k]->delx;
      } else {
	arg.del_embedVec[phraseNum-arg.tgtLen].noalias() = arg.outBufState[k]->delx;
      }
      
      this->embedVecAffine.backward(arg.embedVecEnd[phraseNum-arg.tgtLen], arg.embedVec[phraseNum-arg.tgtLen], 
				    arg.del_embedVec[phraseNum-arg.tgtLen], arg.del_embedVecEnd, grad.embedVecAffineGrad);

      if (rightNum < arg.tgtLen && leftNum < arg.tgtLen) { // word embedding & word embedding
	// tgtRight; parent
	tgtRightNum = data->tgt[rightNum];

	if (grad.targetEmbed.count(tgtRightNum)) {
	  grad.targetEmbed.at(tgtRightNum).noalias() += arg.del_embedVecEnd.segment(0, this->inputDim);
	} else {
	  grad.targetEmbed[tgtRightNum].noalias() = arg.del_embedVecEnd.segment(0, this->inputDim);
	}

	// tgtLeft; child
	tgtLeftNum = data->tgt[leftNum];

	if (grad.targetEmbed.count(tgtLeftNum)) {
	  grad.targetEmbed.at(tgtLeftNum).noalias() += arg.del_embedVecEnd.segment(this->inputDim, this->inputDim);
	} else {
	  grad.targetEmbed[tgtLeftNum].noalias() = arg.del_embedVecEnd.segment(this->inputDim, this->inputDim);
	}

      } else if (rightNum > (arg.tgtLen-1) && leftNum < arg.tgtLen) {
	// tgtRight; parent
	rightNum -= arg.tgtLen;

	if (arg.del_embedVec.count(rightNum)) {
	  arg.del_embedVec[rightNum].noalias() += arg.del_embedVecEnd.segment(0, this->inputDim);
	} else {
	  arg.del_embedVec[rightNum].noalias() = arg.del_embedVecEnd.segment(0, this->inputDim);
	}

	// tgtLeft; child
	tgtLeftNum = data->tgt[leftNum];

	if (grad.targetEmbed.count(tgtLeftNum)) {
	  grad.targetEmbed.at(tgtLeftNum).noalias() += arg.del_embedVecEnd.segment(this->inputDim, this->inputDim);
	} else {
	  grad.targetEmbed[tgtLeftNum].noalias() = arg.del_embedVecEnd.segment(this->inputDim, this->inputDim);
	}

      } else if (rightNum < arg.tgtLen && leftNum > (arg.tgtLen-1)) {
	// tgtRight; parent
	tgtRightNum = data->tgt[rightNum];
	if (grad.targetEmbed.count(tgtRightNum)) {
	  grad.targetEmbed.at(tgtRightNum).noalias() += arg.del_embedVecEnd.segment(0, this->inputDim);
	} else {
	  grad.targetEmbed[tgtRightNum].noalias() = arg.del_embedVecEnd.segment(0, this->inputDim);
	}

	leftNum -= arg.tgtLen;
	// tgtLeft; child
	if (arg.del_embedVec.count(leftNum)) {
	  arg.del_embedVec[leftNum].noalias() += arg.del_embedVecEnd.segment(this->inputDim, this->inputDim);
	} else {
	  arg.del_embedVec[leftNum].noalias() = arg.del_embedVecEnd.segment(this->inputDim, this->inputDim);
	}

      } else {
	// tgtRight; parent
	rightNum -= arg.tgtLen;

	if (arg.del_embedVec.count(rightNum)) {
	  arg.del_embedVec[rightNum].noalias() += arg.del_embedVecEnd.segment(0, this->inputDim);
	} else {
	  arg.del_embedVec[rightNum].noalias() = arg.del_embedVecEnd.segment(0, this->inputDim);
	}

	// tgtLeft; child
	leftNum -= arg.tgtLen;

	if (arg.del_embedVec.count(leftNum)) {
	  arg.del_embedVec[leftNum].noalias() += arg.del_embedVecEnd.segment(this->inputDim, this->inputDim);
	} else {
	  arg.del_embedVec[leftNum].noalias() = arg.del_embedVecEnd.segment(this->inputDim, this->inputDim);
	}
      }

      // act; relation
      if (grad.actionEmbed.count(data->action[i-1])) {
	grad.actionEmbed.at(data->action[i-1]).noalias() += arg.del_embedVecEnd.segment(this->inputDim*2, this->inputActDim);
      } else {
	grad.actionEmbed[data->action[i-1]] = arg.del_embedVecEnd.segment(this->inputDim*2, this->inputActDim);
      }

      --phraseNum;

    } else { // reduce-right
      arg.decState[j]->delh.noalias() += arg.del_utEnd.segment(0, this->hiddenDim);

      top = arg.headList.back();
      arg.headList.pop_back();

       rightNum = arg.embedList.back();
       arg.embedList.pop_back();
       leftNum = arg.embedList.back();
       arg.embedList.pop_back();

       this->outBuf.backward(arg.outBufState[top], arg.outBufState[k], 
			     grad.lstmOutBufGrad, arg.embedVec[phraseNum-arg.tgtLen]);
       if (arg.del_embedVec.count(phraseNum-arg.tgtLen)) {
	 arg.del_embedVec.at(phraseNum-arg.tgtLen).noalias() += arg.outBufState[k]->delx;
       } else {
	 arg.del_embedVec[phraseNum-arg.tgtLen].noalias() = arg.outBufState[k]->delx;
       }

       this->embedVecAffine.backward(arg.embedVecEnd[phraseNum-arg.tgtLen], arg.embedVec[phraseNum-arg.tgtLen], 
				     arg.del_embedVec[phraseNum-arg.tgtLen], arg.del_embedVecEnd, grad.embedVecAffineGrad);

       if (rightNum < arg.tgtLen && leftNum < arg.tgtLen) { // Word Embed & word Embed
	 // tgtRight; child
	 tgtRightNum = data->tgt[rightNum];

	 if (grad.targetEmbed.count(tgtRightNum)) {
	   grad.targetEmbed.at(tgtRightNum).noalias() += arg.del_embedVecEnd.segment(this->inputDim, this->inputDim);
	 } else {
	   grad.targetEmbed[tgtRightNum].noalias() = arg.del_embedVecEnd.segment(this->inputDim, this->inputDim);
	 }

	 // tgtLeft; parent
	 tgtLeftNum = data->tgt[leftNum];

	 if (grad.targetEmbed.count(tgtLeftNum)) {
	   grad.targetEmbed.at(tgtLeftNum).noalias() += arg.del_embedVecEnd.segment(0, this->inputDim);
	 } else {
	   grad.targetEmbed[tgtLeftNum].noalias() = arg.del_embedVecEnd.segment(0, this->inputDim);
	 }

       } else if (rightNum > (arg.tgtLen-1) && leftNum < arg.tgtLen) {
	 // tgtRight; child
	 rightNum -= data->tgt.size();

	 if (arg.del_embedVec.count(rightNum)) {
	   arg.del_embedVec.at(rightNum).noalias() += arg.del_embedVecEnd.segment(this->inputDim, this->inputDim);
	 } else {
	   arg.del_embedVec[rightNum].noalias() = arg.del_embedVecEnd.segment(this->inputDim, this->inputDim);
	 }

	 // tgtLeft; parent
	 tgtLeftNum = data->tgt[leftNum];

	 if (grad.targetEmbed.count(tgtLeftNum)) {
	   grad.targetEmbed.at(tgtLeftNum).noalias() += arg.del_embedVecEnd.segment(0, this->inputDim);
	 } else {
	   grad.targetEmbed[tgtLeftNum].noalias() = arg.del_embedVecEnd.segment(0, this->inputDim);
	 }

       } else if (rightNum < arg.tgtLen && leftNum > (arg.tgtLen-1)) {
	 // tgtRight; child
	 tgtRightNum = data->tgt[rightNum];

	 if (grad.targetEmbed.count(tgtRightNum)) {
	   grad.targetEmbed.at(tgtRightNum).noalias() += arg.del_embedVecEnd.segment(this->inputDim, this->inputDim);
	 } else {
	   grad.targetEmbed[tgtRightNum].noalias() = arg.del_embedVecEnd.segment(this->inputDim, this->inputDim);
	 }

	 // tgtLeft; parent
	 leftNum -= arg.tgtLen;

	 if (arg.del_embedVec.count(leftNum)) {
	   arg.del_embedVec[leftNum].noalias() += arg.del_embedVecEnd.segment(0, this->inputDim);
	 } else {
	   arg.del_embedVec[leftNum].noalias() = arg.del_embedVecEnd.segment(0, this->inputDim);
	 }

       } else {
	 // tgtRight; child
	 rightNum -= data->tgt.size();

	 if (arg.del_embedVec.count(rightNum)) {
	   arg.del_embedVec[rightNum].noalias() += arg.del_embedVecEnd.segment(this->inputDim, this->inputDim);
	 } else {
	   arg.del_embedVec[rightNum].noalias() = arg.del_embedVecEnd.segment(this->inputDim, this->inputDim);
	 }

	 // tgtLeft; parent
	 leftNum -= data->tgt.size();

	 if (arg.del_embedVec.count(leftNum)) {
	   arg.del_embedVec[leftNum].noalias() += arg.del_embedVecEnd.segment(0, this->inputDim);
	 } else {
	   arg.del_embedVec[leftNum].noalias() = arg.del_embedVecEnd.segment(0, this->inputDim);
	 }
       }

       // act; relation
       if (grad.actionEmbed.count(data->action[i-1])) {
	 grad.actionEmbed.at(data->action[i-1]).noalias() += arg.del_embedVecEnd.segment(this->inputDim*2, this->inputActDim);
       } else {
	 grad.actionEmbed[data->action[i-1]].noalias() = arg.del_embedVecEnd.segment(this->inputDim*2, this->inputActDim);
       }

       --phraseNum;
     }
     if (i > 0) {
       // Backward
       this->act.backward1(arg.actState[i-1], arg.actState[i], 
			   grad.lstmActGrad, this->actionEmbed.col(data->action[i-1]));
       if (grad.actionEmbed.count(data->action[i-1])) {
	 grad.actionEmbed.at(data->action[i-1]) += arg.actState[i]->delx;
       } else {
	 grad.actionEmbed[data->action[i-1]] = arg.actState[i]->delx;
       }
     } else {}
   }

   // Affine (this->utAffine)
   for (int d = 0, D = arg.utEnd[0].rows(); d < D; ++d){
     for (int i = 0; i < arg.actLen; ++i){
       this->utAffine.backward2(arg.utEnd[i], arg.del_ut[i], d, grad.utAffineGrad);
     }
   }
   for (int d = 0, D = arg.stildeEnd[0].rows(); d < D; ++d){
     for (int i = 0; i < arg.tgtLen; ++i){
       this->stildeAffine.backward2(arg.stildeEnd[i], arg.del_stilde[i], d, grad.stildeAffineGrad);
     }
   }

   this->decoderActionBackward2(data, arg, grad);
   this->decoderBackward2(data, arg, grad);

   // Decoder -> Encoder
   this->decInitAffine.backward(arg.encStateEnd, arg.decState[0]->h, arg.decState[0]->delh, 
				arg.del_decInitStateEnd, grad.decInitAffineGrad);
   arg.encState[arg.srcLen-1]->delh.noalias() += arg.del_decInitStateEnd.segment(0, this->hiddenEncDim);
   arg.encRevState[arg.srcLen-1]->delh.noalias() += arg.del_decInitStateEnd.segment(this->hiddenEncDim, this->hiddenEncDim);

   // Act -> Encoder
   this->actInitAffine.backward(arg.encStateEnd, arg.actState[0]->h, arg.actState[0]->delh, 
				arg.del_encStateEnd, grad.actInitAffineGrad);
   arg.encState[arg.srcLen-1]->delh.noalias() += arg.del_encStateEnd.segment(0, this->hiddenEncDim);
   arg.encRevState[arg.srcLen-1]->delh.noalias() += arg.del_encStateEnd.segment(this->hiddenEncDim, this->hiddenEncDim);

   // Output Buffer -> Encoder
   this->outBufInitAffine.backward(arg.encStateEnd, arg.outBufState[0]->h, arg.outBufState[0]->delh, 
				   arg.del_outBufInitStateEnd, grad.outBufInitAffineGrad);
   arg.encState[arg.srcLen-1]->delh.noalias() += arg.del_outBufInitStateEnd.segment(0, this->hiddenEncDim);
   arg.encRevState[arg.srcLen-1]->delh.noalias() += arg.del_outBufInitStateEnd.segment(this->hiddenEncDim, this->hiddenEncDim);

   for (int i = arg.srcLen-1; i > 0; --i) {
     this->enc.backward1(arg.encState[i-1], arg.encState[i], grad.lstmSrcGrad, 
			 this->sourceEmbed.col(data->src[i]));
     this->encRev.backward1(arg.encRevState[i-1], arg.encRevState[i], grad.lstmSrcRevGrad, 
			    this->sourceEmbed.col(data->src[arg.srcLen-i-1]));
     if (grad.sourceEmbed.count(data->src[i])) {
       grad.sourceEmbed.at(data->src[i]).noalias() += arg.encState[i]->delx;
     } else {
       grad.sourceEmbed[data->src[i]].noalias() = arg.encState[i]->delx;
     }
     if (grad.sourceEmbed.count(data->src[arg.srcLen-i-1])) {
       grad.sourceEmbed.at(data->src[arg.srcLen-i-1]).noalias() += arg.encRevState[i]->delx;
     } else {
       grad.sourceEmbed[data->src[arg.srcLen-i-1]].noalias() = arg.encRevState[i]->delx;
     }
   }
   // i == 0
   this->enc.backward1(arg.encState[0], grad.lstmSrcGrad, 
		       this->sourceEmbed.col(data->src[0]));
   this->encRev.backward1(arg.encRevState[0], grad.lstmSrcRevGrad, 
			  this->sourceEmbed.col(data->src[length])); // length+1-1 = srcLen -i
   if (grad.sourceEmbed.count(data->src[0])) {
     grad.sourceEmbed.at(data->src[0]).noalias() += arg.encState[0]->delx;
   } else {
     grad.sourceEmbed[data->src[0]].noalias() = arg.encState[0]->delx;
   }
   if (grad.sourceEmbed.count(data->src[arg.srcLen-1])) {
     grad.sourceEmbed.at(data->src[arg.srcLen-1]).noalias() += arg.encRevState[0]->delx;
   } else {
     grad.sourceEmbed[data->src[arg.srcLen-1]].noalias() = arg.encRevState[0]->delx;
   }
  this->biEncoderBackward2(data, arg, grad);
  arg.clear();
}

void NMTRNNG::calculateAlpha(NMTRNNG::ThreadArg& arg, 
			     const LSTM::State* decState) { // calculate attentional weight;
  
  for (int i = 0; i < arg.srcLen; ++i) {
    arg.alphaSeqVec.coeffRef(i, 0) = decState->h.dot(this->Wgeneral * arg.biEncState[i]);
  }

  // softmax of ``alphaSeq``
  arg.alphaSeqVec.array() -= arg.alphaSeqVec.maxCoeff(); // stable softmax
  arg.alphaSeqVec = arg.alphaSeqVec.array().exp(); // exp() operation for all elements; np.exp(alphaSeq) 
  arg.alphaSeqVec /= arg.alphaSeqVec.array().sum(); // alphaSeq.sum()
}

void NMTRNNG::calculateAlpha(NMTRNNG::ThreadArg& arg, 
			     const LSTM::State* decState,
			     const int colNum) { // calculate attentional weight;
  for (int i = 0; i < arg.srcLen; ++i) {
    arg.alphaSeq.coeffRef(i, colNum) = decState->h.dot(this->Wgeneral * arg.biEncState[i]);
  }
  // softmax of ``alphaSeq``
  arg.alphaSeq.col(colNum).array() -= arg.alphaSeq.col(colNum).maxCoeff(); // stable softmax
  arg.alphaSeq.col(colNum) = arg.alphaSeq.col(colNum).array().exp(); // exp() operation for all elements; np.exp(alphaSeq) 
  arg.alphaSeq.col(colNum) /= arg.alphaSeq.col(colNum).array().sum(); // alphaSeq.sum()
}

bool NMTRNNG::trainOpenMP(NMTRNNG::Grad& grad) {
  static std::vector<NMTRNNG::ThreadArg> args;
  static std::vector<std::pair<int, int> > miniBatch;
  static std::vector<NMTRNNG::Grad> grads;

  Real lossTrain = 0.0;
  // Real lossActTrain = 0.0;
  Real lossDev = 0.0;
  Real lossActDev = 0.0;
  Real tgtNum = 0.0;
  Real actNum = 0.0;
  Real gradNorm;
  Real lr = this->learningRate;
  static float countModel = this->startIter-0.5;

  float countModelTmp = countModel;
  std::string modelFileNameTmp = "";

  if (args.empty()) {
    grad = NMTRNNG::Grad(*this);
    for (int i = 0; i < this->threadNum; ++i) {
      args.push_back(NMTRNNG::ThreadArg(*this));
      grads.push_back(NMTRNNG::Grad(*this));
    }
    for (int i = 0, step = this->trainData.size()/this->miniBatchSize; i< step; ++i) {
      miniBatch.push_back(std::pair<int, int>(i*this->miniBatchSize, (i == step-1 ? this->trainData.size()-1 : (i+1)*this->miniBatchSize-1)));
    }
  }

  auto startTmp = std::chrono::system_clock::now();
  this->rnd.shuffle(this->trainData);

  int count = 0;
  for (auto it = miniBatch.begin(); it != miniBatch.end(); ++it) {
    std::cout << "\r"
	      << "Progress: " << ++count << "/" << miniBatch.size() << " mini batches" << std::flush;
    for (auto it = args.begin(); it != args.end(); ++it) {
      it->initLoss();
    }
#pragma omp parallel for num_threads(this->threadNum) schedule(dynamic) shared(args, grad, grads)
    for (int i = it->first; i <= it->second; ++i) {
      int id = omp_get_thread_num();

      this->train(this->trainData[i], args[id], grads[id]);

      /* ..Gradient Checking.. :) */
      // this->gradientChecking(this->trainData[i], args[id], grads[id]);
    }
    for (int id = 0; id < this->threadNum; ++id) {
      grad += grads[id];
      grads[id].init();
      lossTrain += args[id].loss; // loss + lossAct
      args[id].loss = 0.0;
    }

    gradNorm = sqrt(grad.norm())/this->miniBatchSize;
    if (Utils::infNan2(gradNorm)) {
      countModel = countModelTmp;

      grad.init();
      std::cout << "(!) Error: INF/ NAN Gradients. Resume the training." << std::endl;
      
      if (modelFileNameTmp != "") {
	int systemVal = system(((std::string)"rm " + modelFileNameTmp).c_str());
	if (systemVal == -1) {
	  std::cout << "Fails to remove "
		    << modelFileNameTmp.c_str() << std::endl;
	}
      }
      return false;
    }
    lr = (gradNorm > this->clipThreshold ? this->clipThreshold*this->learningRate/gradNorm : this->learningRate);
    lr /= this->miniBatchSize;

    if (this->opt == NMTRNNG::SGD) {
      // Update the gradients by SGD
      grad.sgd(*this, lr);
    }
    grad.init();
      
    if (count == ((int)miniBatch.size()/2)) { // saveModel after halving epoch
      this->saveModel(grad, countModel);
      countModel += 1.;
    }

  }

  // Save a model
  std::string currentModelFileName;
  std::string currentGradFileName;
  std::tie(currentModelFileName, currentGradFileName) = this->saveModel(grad, countModel-0.5);

  std::cout << std::endl;
  auto endTmp = std::chrono::system_clock::now();
  std::cout << "Training time for this epoch: " 
	    << (std::chrono::duration_cast<std::chrono::seconds>(endTmp-startTmp).count())/60.0 << "min." << std::endl;
  std::cout << "Training Loss (/sentence):    " 
	    << lossTrain/this->trainData.size() << std::endl;

  startTmp = std::chrono::system_clock::now();
#pragma omp parallel for num_threads(this->threadNum)
  for (int i = 0; i < (int)this->devData.size(); ++i) {
    Real loss, lossAct;
    int id = omp_get_thread_num();
    std::tie(loss, lossAct) = this->calcLoss(this->devData[i], args[id], false);
#pragma omp critical
    {
      lossDev += loss;
      lossActDev += lossAct;
      tgtNum += this->devData[i]->tgt.size();
      actNum += this->devData[i]->action.size();
    }
  }

  endTmp = std::chrono::system_clock::now();

  std::cout << "Evaluation time for this epoch: " 
	    << (std::chrono::duration_cast<std::chrono::seconds>(endTmp-startTmp).count())/60.0 
	    << "min." 
	    << std::endl;
  std::cout << "[Language] Development Perplexity and Loss (/sentence):  " 
	    << exp(lossDev/tgtNum) << ", "
	    << lossDev/this->devData.size() << std::endl;
  std::cout << "[Action] Development Perplexity and Loss (/sentence):  " 
	    << exp(lossActDev/actNum) << ", "
	    << lossActDev/this->devData.size() << std::endl;

  Real devPerp = exp(lossDev/tgtNum);
  if (this->prevPerp < devPerp){
    countModel = countModelTmp;
    std::cout << "(!) Notes: [Language] Dev perplexity became worse, Resume the training!" << std::endl;

    // system(((std::string)"rm "+modelFileNameTmp).c_str());
    // system(((std::string)"rm "+currentModelFileName).c_str());

    return false;
  }
  this->prevPerp = devPerp;
  this->prevModelFileName = currentModelFileName;
  this->prevGradFileName = currentGradFileName;

  saveResult(lossTrain/this->trainData.size(), ".trainLoss"); // Training Loss
  saveResult(exp(lossDev/tgtNum), ".devPerp");                // [Language] Perplexity
  saveResult(exp(lossActDev/actNum), ".devActPerp");          // [Action]   Perplexity
  saveResult(lossDev/this->devData.size(), ".devLoss");       // Development Loss
  
  return true;
}

std::tuple<Real, Real> NMTRNNG::calcLoss(NMTRNNG::Data* data, 
					 NMTRNNG::ThreadArg& arg, 
					 const bool train) {
  Real loss = 0.0;
  Real lossAct = 0.0;
  int j = 0;
  int k = 0;
  int phraseNum = data->tgt.size();
  int actNum;

  arg.init(*this, data, false);
  this->biEncode(data, arg, false);

  // k == 0
  this->outBufInitAffine.forward(arg.encStateEnd, arg.outBufState[k]->h);
  arg.outBufState[k]->c = this->zeros;

  arg.headStack.push(k);
  ++k;

  for (int i = 0; i < arg.actLen; ++i, ++k) {
    // SoftmaxAct calculation
    actNum = data->action[i];

    this->decoderAction(arg, arg.actState, data->action[i-1], i, false); // PUSH

    if (this->actionVoc.tokenList[actNum]->action == 0) { // 0: Shift
      arg.headStack.push(k);
      // 1) Let a decoder proceed one step; PUSH
      this->decoder(arg, arg.decState, arg.s_tilde[j-1], data->tgt[j-1], j, false);
      /* Attention */
      this->decoderAttention(arg, arg.decState[j], arg.contextSeqList[j], arg.s_tilde[j], arg.stildeEnd[j]);
      if (!this->useBlackOut) {
	this->softmax.calcDist(arg.s_tilde[j], arg.targetDist);
	loss += this->softmax.calcLoss(arg.targetDist, data->tgt[j]);
      } else {
	if (train) {
	  // word prediction
	  arg.blackOutState[0].sample[0] = data->tgt[j];
	  arg.blackOutState[0].weight.col(0) = this->blackOut.weight.col(data->tgt[j]);
	  arg.blackOutState[0].bias.coeffRef(0, 0) = this->blackOut.bias.coeff(data->tgt[j], 0);

	  this->blackOut.calcSampledDist2(arg.s_tilde[j], arg.targetDist, arg.blackOutState[0]);
	  loss += this->blackOut.calcSampledLoss(arg.targetDist); // Softmax
	} else { // Test Time
	  this->blackOut.calcDist(arg.s_tilde[j], arg.targetDist); // Softmax
	  loss += this->blackOut.calcLoss(arg.targetDist, data->tgt[j]); // Softmax
	}
      }
      // 2) Let the output buffer proceed one step, though the computed unit is not used at this step; PUSH
      this->outBuf.forward(this->targetEmbed.col(data->tgt[j]), 
			   arg.outBufState[k-1], arg.outBufState[k]);

      arg.embedStack.push(j);

      // SoftmaxAct calculation (o: output buffer, s: stack, and h: action)
      arg.utEnd[0].segment(0, this->hiddenDim).noalias() = arg.decState[j]->h;
      ++j;

    } else if (this->actionVoc.tokenList[actNum]->action == 1) { // 1: Reduce-Left
      this->decoderReduceLeft(data, arg, phraseNum, i-1, k, false);
      ++phraseNum;

      // SoftmaxAct calculation (o: output buffer, s: stack, and h: action)
      arg.utEnd[0].segment(0, this->hiddenDim).noalias() = arg.decState[j-1]->h;

    } else if (this->actionVoc.tokenList[actNum]->action == 2) { // 2: Reduce-Right
      this->decoderReduceRight(data, arg, phraseNum, i-1, k, false);
      ++phraseNum;

      // SoftmaxAct calculation (o: output buffer, s: stack, and h: action)
      arg.utEnd[0].segment(0, this->hiddenDim).noalias() = arg.decState[j-1]->h;

    } else {
      print("Error Non-Shift/Reduce");
      exit(2);
    }
    arg.utEnd[0].segment(this->hiddenDim, this->hiddenDim).noalias() = arg.outBufState[k-1]->h; 
    arg.utEnd[0].segment(this->hiddenDim*2, this->hiddenActDim).noalias() = arg.actState[i]->h; 
    this->utAffine.forward(arg.utEnd[0], arg.ut[0]);

    this->softmaxAct.calcDist(arg.ut[0], arg.actionDist);
    lossAct += this->softmaxAct.calcLoss(arg.actionDist, data->action[i]);
  }

  arg.clear();
  return std::forward_as_tuple(loss, lossAct);
}

void NMTRNNG::gradientChecking(NMTRNNG::Data* data, 
			       NMTRNNG::ThreadArg& arg,
			       NMTRNNG::Grad& grad) {
      print("--Softmax");
      if (!this->useBlackOut) {
	print(" softmax_W");
	this->gradChecker(data, arg, this->softmax.weight, grad.softmaxGrad.weight);
	print(" softmax_b");
	this->gradChecker(data, arg, this->softmax.bias, grad.softmaxGrad.bias);
      }
      print(" softmaxAct_W");
      this->gradChecker(data, arg, this->softmaxAct.weight, grad.softmaxActGrad.weight);
      print(" softmaxAct_b");
      this->gradChecker(data, arg, this->softmaxAct.bias, grad.softmaxActGrad.bias);

      // Decoder
      print("--Decoder");
      print(" utAffine_W");
      this->gradChecker(data, arg, this->utAffine.weight, grad.utAffineGrad.weightGrad);
      print(" utAffine_b");
      this->gradChecker(data, arg, this->utAffine.bias, grad.utAffineGrad.biasGrad);

      print(" embedVecAffine_W");
      this->gradChecker(data, arg, this->embedVecAffine.weight, grad.embedVecAffineGrad.weightGrad);
      print(" embedVecAffine_b");
      this->gradChecker(data, arg, this->embedVecAffine.bias, grad.embedVecAffineGrad.biasGrad);

      print(" stildeAffine_W");
      this->gradChecker(data, arg, this->stildeAffine.weight, grad.stildeAffineGrad.weightGrad);
      print(" stildeAffine_b");
      this->gradChecker(data, arg, this->stildeAffine.bias, grad.stildeAffineGrad.biasGrad);

      print(" dec_Wx");
      this->gradChecker(data, arg, this->dec.Wxi, grad.lstmTgtGrad.Wxi);
      this->gradChecker(data, arg, this->dec.Wxf, grad.lstmTgtGrad.Wxf);
      this->gradChecker(data, arg, this->dec.Wxo, grad.lstmTgtGrad.Wxo);
      this->gradChecker(data, arg, this->dec.Wxu, grad.lstmTgtGrad.Wxu);
      print(" dec_Wh");
      this->gradChecker(data, arg, this->dec.Whi, grad.lstmTgtGrad.Whi);
      this->gradChecker(data, arg, this->dec.Whf, grad.lstmTgtGrad.Whf);
      this->gradChecker(data, arg, this->dec.Who, grad.lstmTgtGrad.Who);
      this->gradChecker(data, arg, this->dec.Whu, grad.lstmTgtGrad.Whu);
      print(" dec_Wa");
      this->gradChecker(data, arg, this->dec.Wai, grad.lstmTgtGrad.Wai);
      this->gradChecker(data, arg, this->dec.Waf, grad.lstmTgtGrad.Waf);
      this->gradChecker(data, arg, this->dec.Wao, grad.lstmTgtGrad.Wao);
      this->gradChecker(data, arg, this->dec.Wau, grad.lstmTgtGrad.Wau);
      print(" dec_b");
      this->gradChecker(data, arg, this->dec.bi, grad.lstmTgtGrad.bi);
      this->gradChecker(data, arg, this->dec.bf, grad.lstmTgtGrad.bf);
      this->gradChecker(data, arg, this->dec.bo, grad.lstmTgtGrad.bo);
      this->gradChecker(data, arg, this->dec.bu, grad.lstmTgtGrad.bu);

      print(" act_Wx");
      this->gradChecker(data, arg, this->act.Wxi, grad.lstmActGrad.Wxi);
      this->gradChecker(data, arg, this->act.Wxf, grad.lstmActGrad.Wxf);
      this->gradChecker(data, arg, this->act.Wxo, grad.lstmActGrad.Wxo);
      this->gradChecker(data, arg, this->act.Wxu, grad.lstmActGrad.Wxu);
      print(" act_Wh");
      this->gradChecker(data, arg, this->act.Whi, grad.lstmActGrad.Whi);
      this->gradChecker(data, arg, this->act.Whf, grad.lstmActGrad.Whf);
      this->gradChecker(data, arg, this->act.Who, grad.lstmActGrad.Who);
      this->gradChecker(data, arg, this->act.Whu, grad.lstmActGrad.Whu);
      print(" act_b");
      this->gradChecker(data, arg, this->act.bi, grad.lstmActGrad.bi);
      this->gradChecker(data, arg, this->act.bf, grad.lstmActGrad.bf);
      this->gradChecker(data, arg, this->act.bo, grad.lstmActGrad.bo);
      this->gradChecker(data, arg, this->act.bu, grad.lstmActGrad.bu);

      print(" outBuf_Wx");
      this->gradChecker(data, arg, this->outBuf.Wxi, grad.lstmOutBufGrad.Wxi);
      this->gradChecker(data, arg, this->outBuf.Wxf, grad.lstmOutBufGrad.Wxf);
      this->gradChecker(data, arg, this->outBuf.Wxo, grad.lstmOutBufGrad.Wxo);
      this->gradChecker(data, arg, this->outBuf.Wxu, grad.lstmOutBufGrad.Wxu);
      print(" outBuf_Wh");
      this->gradChecker(data, arg, this->outBuf.Whi, grad.lstmOutBufGrad.Whi);
      this->gradChecker(data, arg, this->outBuf.Whf, grad.lstmOutBufGrad.Whf);
      this->gradChecker(data, arg, this->outBuf.Who, grad.lstmOutBufGrad.Who);
      this->gradChecker(data, arg, this->outBuf.Whu, grad.lstmOutBufGrad.Whu);
      print(" outBuf_b");
      this->gradChecker(data, arg, this->outBuf.bi, grad.lstmOutBufGrad.bi);
      this->gradChecker(data, arg, this->outBuf.bf, grad.lstmOutBufGrad.bf);
      this->gradChecker(data, arg, this->outBuf.bo, grad.lstmOutBufGrad.bo);
      this->gradChecker(data, arg, this->outBuf.bu, grad.lstmOutBufGrad.bu);

      print(" Wgeneral");
      this->gradChecker(data, arg, this->Wgeneral, grad.Wgeneral);

      print("--Initial Decoder");
      print(" decInitAffine_W");
      this->gradChecker(data, arg, this->decInitAffine.weight, grad.decInitAffineGrad.weightGrad);
      print(" decInitAffine_b");
      this->gradChecker(data, arg, this->decInitAffine.bias, grad.decInitAffineGrad.biasGrad);

      print(" actInitAffine_W");
      this->gradChecker(data, arg, this->actInitAffine.weight, grad.actInitAffineGrad.weightGrad);
      print(" actInitAffine_b");
      this->gradChecker(data, arg, this->actInitAffine.bias, grad.actInitAffineGrad.biasGrad);

      print(" outBufInitAffine_W");
      this->gradChecker(data, arg, this->outBufInitAffine.weight, grad.outBufInitAffineGrad.weightGrad);
      print(" actInitAffine_b");
      this->gradChecker(data, arg, this->outBufInitAffine.bias, grad.outBufInitAffineGrad.biasGrad);

      // Encoder
      print("--Encoder");
      print(" enc_Wx");
      this->gradChecker(data, arg, this->enc.Wxi, grad.lstmSrcGrad.Wxi);
      this->gradChecker(data, arg, this->enc.Wxf, grad.lstmSrcGrad.Wxf);
      this->gradChecker(data, arg, this->enc.Wxo, grad.lstmSrcGrad.Wxo);
      this->gradChecker(data, arg, this->enc.Wxu, grad.lstmSrcGrad.Wxu);
      print(" enc_Wh");
      this->gradChecker(data, arg, this->enc.Whi, grad.lstmSrcGrad.Whi);
      this->gradChecker(data, arg, this->enc.Whf, grad.lstmSrcGrad.Whf);
      this->gradChecker(data, arg, this->enc.Who, grad.lstmSrcGrad.Who);
      this->gradChecker(data, arg, this->enc.Whu, grad.lstmSrcGrad.Whu);
      print(" enc_b");
      this->gradChecker(data, arg, this->enc.bi, grad.lstmSrcGrad.bi);
      this->gradChecker(data, arg, this->enc.bf, grad.lstmSrcGrad.bf);
      this->gradChecker(data, arg, this->enc.bo, grad.lstmSrcGrad.bo);
      this->gradChecker(data, arg, this->enc.bu, grad.lstmSrcGrad.bu);

      print(" encRev_Wx");
      this->gradChecker(data, arg, this->encRev.Wxi, grad.lstmSrcRevGrad.Wxi);
      this->gradChecker(data, arg, this->encRev.Wxf, grad.lstmSrcRevGrad.Wxf);
      this->gradChecker(data, arg, this->encRev.Wxo, grad.lstmSrcRevGrad.Wxo);
      this->gradChecker(data, arg, this->encRev.Wxu, grad.lstmSrcRevGrad.Wxu);
      print(" encRev_Wh");
      this->gradChecker(data, arg, this->encRev.Whi, grad.lstmSrcRevGrad.Whi);
      this->gradChecker(data, arg, this->encRev.Whf, grad.lstmSrcRevGrad.Whf);
      this->gradChecker(data, arg, this->encRev.Who, grad.lstmSrcRevGrad.Who);
      this->gradChecker(data, arg, this->encRev.Whu, grad.lstmSrcRevGrad.Whu);
      print(" encRev_b");
      this->gradChecker(data, arg, this->encRev.bi, grad.lstmSrcRevGrad.bi);
      this->gradChecker(data, arg, this->encRev.bf, grad.lstmSrcRevGrad.bf);
      this->gradChecker(data, arg, this->encRev.bo, grad.lstmSrcRevGrad.bo);
      this->gradChecker(data, arg, this->encRev.bu, grad.lstmSrcRevGrad.bu);

      // Embeddings
      print("--Embeddings");
      print(" sourceEmbed; targetEmbed; actionEmbed");
      this->gradChecker(data, arg, grad);
}

void NMTRNNG::gradChecker(NMTRNNG::Data* data, 
			  NMTRNNG::ThreadArg& arg,
			  MatD& param, 
			  const MatD& grad) {
  const Real EPS = 1.0e-04;

  for (int i = 0; i < param.rows(); ++i) {
    for (int j = 0; j < param.cols(); ++j) {
      Real val= 0.0;
      Real objFuncWordPlus = 0.0;
      Real objFuncActPlus = 0.0;
      Real objFuncWordMinus = 0.0;
      Real objFuncActMinus = 0.0;
      val = param.coeff(i, j); // Θ_i
      param.coeffRef(i, j) = val + EPS;
      std::tie(objFuncWordPlus, objFuncActPlus) = this->calcLoss(data, arg, true);
      param.coeffRef(i, j) = val - EPS;
      std::tie(objFuncWordMinus, objFuncActMinus) = this->calcLoss(data, arg, true);
      param.coeffRef(i, j) = val;

      Real gradVal = grad.coeff(i, j);
      Real enumVal = ((objFuncWordPlus + objFuncActPlus) - (objFuncWordMinus + objFuncActMinus))/ (2.0*EPS);
      if ((gradVal - enumVal) > 1.0e-05) {
	std::cout << "Grad: " << gradVal << std::endl;
	std::cout << "Enum: " << enumVal << std::endl;
      } else {}
    }
  }
}

void NMTRNNG::gradChecker(NMTRNNG::Data* data,
			  NMTRNNG::ThreadArg& arg, 
			  VecD& param, 
			  const MatD& grad) {
  const Real EPS = 1.0e-04;
 
  for (int i = 0; i < param.rows(); ++i) {
    Real val= 0.0;
    Real objFuncWordPlus = 0.0;
    Real objFuncActPlus = 0.0;
    Real objFuncWordMinus = 0.0;
    Real objFuncActMinus = 0.0;
    val = param.coeff(i, 0); // Θ_i
    param.coeffRef(i, 0) = val + EPS;
    std::tie(objFuncWordPlus, objFuncActPlus) = this->calcLoss(data, arg, true);      
    param.coeffRef(i, 0) = val - EPS;
    std::tie(objFuncWordMinus, objFuncActMinus) = this->calcLoss(data, arg, true);
    param.coeffRef(i, 0) = val;
 
    Real gradVal = grad.coeff(i, 0);
    Real enumVal = ((objFuncWordPlus + objFuncActPlus) - (objFuncWordMinus + objFuncActMinus))/ (2.0*EPS);
    if ((gradVal - enumVal) > 1.0e-05) {
      std::cout << "Grad: " << gradVal << std::endl;
      std::cout << "Enum: " << enumVal << std::endl ;
    } else {}
  }
}

void NMTRNNG::gradChecker(NMTRNNG::Data* data, 
			  NMTRNNG::ThreadArg& arg, 
			  NMTRNNG::Grad& grad) {
  const Real EPS = 1.0e-04;

  for (auto it = grad.sourceEmbed.begin(); it != grad.sourceEmbed.end(); ++it) {
    for (int i = 0; i < it->second.rows(); ++i) {
      Real val = 0.0;
      Real objFuncWordPlus = 0.0;
      Real objFuncActPlus = 0.0;
      Real objFuncWordMinus = 0.0;
      Real objFuncActMinus = 0.0;
      val = this->sourceEmbed.coeff(i, it->first); // Θ_i
      this->sourceEmbed.coeffRef(i, it->first) = val + EPS;
      std::tie(objFuncWordPlus, objFuncActPlus) = this->calcLoss(data, arg, true);
      this->sourceEmbed.coeffRef(i, it->first) = val - EPS;
      std::tie(objFuncWordMinus, objFuncActMinus) = this->calcLoss(data, arg, true);
      this->sourceEmbed.coeffRef(i, it->first) = val;

      Real gradVal = it->second.coeff(i, 0);
      Real enumVal = ((objFuncWordPlus + objFuncActPlus) - (objFuncWordMinus + objFuncActMinus))/ (2.0*EPS);
      if ((gradVal - enumVal) > 1.0e-05) {
	std::cout << "Grad: " << gradVal << std::endl;
	std::cout << "Enum: " << enumVal << std::endl ;
      } else {}
    }
  }

  for (auto it = grad.targetEmbed.begin(); it != grad.targetEmbed.end(); ++it) {
    for (int i = 0; i < it->second.rows(); ++i) {
      Real val= 0.0;
      Real objFuncWordPlus = 0.0;
      Real objFuncActPlus = 0.0;
      Real objFuncWordMinus = 0.0;
      Real objFuncActMinus = 0.0;
      val = this->targetEmbed.coeff(i, it->first); // Θ_i
      this->targetEmbed.coeffRef(i, it->first) = val + EPS;
      std::tie(objFuncWordPlus, objFuncActPlus) = this->calcLoss(data, arg, true);
      this->targetEmbed.coeffRef(i, it->first) = val - EPS;
      std::tie(objFuncWordMinus, objFuncActMinus) = this->calcLoss(data, arg, true);
      this->targetEmbed.coeffRef(i, it->first) = val;

      Real gradVal = it->second.coeff(i, 0);
      Real enumVal = ((objFuncWordPlus + objFuncActPlus) - (objFuncWordMinus + objFuncActMinus))/ (2.0*EPS);
      if ((gradVal - enumVal) > 1.0e-05) {
	std::cout << "Grad: " << gradVal << std::endl;
	std::cout << "Enum: " << enumVal << std::endl ;
      } else {}
    }
  }
  std::cout << std::endl;
  for (auto it = grad.actionEmbed.begin(); it != grad.actionEmbed.end(); ++it) {
    for (int i = 0; i < it->second.rows(); ++i) {
      Real val= 0.0;
      Real objFuncWordPlus = 0.0;
      Real objFuncActPlus = 0.0;
      Real objFuncWordMinus = 0.0;
      Real objFuncActMinus = 0.0;
      val = this->actionEmbed.coeff(i, it->first); // Θ_i
      this->actionEmbed.coeffRef(i, it->first) = val + EPS;
      std::tie(objFuncWordPlus, objFuncActPlus) = this->calcLoss(data, arg, true);
      this->actionEmbed.coeffRef(i, it->first) = val - EPS;
      std::tie(objFuncWordMinus, objFuncActMinus) = this->calcLoss(data, arg, true);
      this->actionEmbed.coeffRef(i, it->first) = val;

      Real gradVal = it->second.coeff(i, 0);
      Real enumVal = ((objFuncWordPlus + objFuncActPlus) - (objFuncWordMinus + objFuncActMinus))/ (2.0*EPS);
      if ((gradVal - enumVal) > 1.0e-05) {
	std::cout << "Grad: " << gradVal << std::endl;
	std::cout << "Enum: " << enumVal << std::endl ;
      } else {}
    }
  }
}

void NMTRNNG::makeTrans(const std::vector<int>& tgt, 
			std::vector<int>& trans) {
  for (auto it = tgt.begin(); it != tgt.end(); ++it) {
    if (*it != this->targetVoc.eosIndex) {
      trans.push_back(*it);
    } else {}
  }
}

void NMTRNNG::loadCorpus(const std::string& src, 
			 const std::string& tgt, 
			 const std::string& act,
			 std::vector<NMTRNNG::Data*>& data) {
  std::ifstream ifsSrc(src.c_str());
  std::ifstream ifsTgt(tgt.c_str());
  std::ifstream ifsAct(act.c_str());

  assert(ifsSrc);
  assert(ifsTgt);
  assert(ifsAct);

  int numLine = 0;
  // Src
  for (std::string line; std::getline(ifsSrc, line);) {
    std::vector<std::string> tokens;
    NMTRNNG::Data *dataTmp(NULL);
    dataTmp = new NMTRNNG::Data;
    data.push_back(dataTmp);
    Utils::split(line, tokens);

    for (auto it = tokens.begin(); it != tokens.end(); ++it) {
      data.back()->src.push_back(sourceVoc.tokenIndex.count(*it) ? sourceVoc.tokenIndex.at(*it) : sourceVoc.unkIndex);
    }
    data.back()->src.push_back(sourceVoc.eosIndex); // EOS
  }

  // Tgt
  for (std::string line; std::getline(ifsTgt, line);) {
    std::vector<std::string> tokens;

    Utils::split(line, tokens);

    for (auto it = tokens.begin(); it != tokens.end(); ++it) {
      data[numLine]->tgt.push_back(targetVoc.tokenIndex.count(*it) ? targetVoc.tokenIndex.at(*it) : targetVoc.unkIndex);
    }
    data[numLine]->tgt.push_back(targetVoc.eosIndex); // EOS
    ++numLine;
  }

  numLine = 0;
  // Action
  for (std::string line; std::getline(ifsAct, line);) {
    std::vector<std::string> tokens;
    Utils::split(line, tokens);
    if (!tokens.empty()) {
       if (actionVoc.tokenIndex.count(tokens[0])) {
	data[numLine]->action.push_back(actionVoc.tokenIndex.at(tokens[0]));
      } else {
	 print("Error: Unknown word except shift/reduce.");
	 exit(1);
      }
     } else {
      ++numLine;
    }
  }
}

std::tuple<std::string, std::string> NMTRNNG::saveModel(NMTRNNG::Grad& grad, 
							const float i) {
  std::ostringstream oss;
  oss << this->saveDirName << "Model_NMTRNNG"
      << ".itr_" << i+1
      << ".BlackOut_" << (this->useBlackOut?"true":"false")
      << ".beamSize_" << this->beamSize 
      << ".miniBatchSize_" << this->miniBatchSize
      << ".threadNum_" << this->threadNum
      << ".lrSGD_"<< this->learningRate 
      << ".bin"; 
  this->save(oss.str());

  std::ostringstream ossGrad;
  ossGrad << this->saveDirName << "Model_NMTRNNGGrad"
	  << ".itr_" << i+1
	  << ".BlackOut_" << (this->useBlackOut?"true":"false")
	  << ".beamSize_" << this->beamSize 
	  << ".miniBatchSize_" << this->miniBatchSize
	  << ".threadNum_" << this->threadNum
	  << ".lrSGD_"<< this->learningRate 
	  << ".bin"; 

  return std::forward_as_tuple(oss.str(), ossGrad.str());
}

void NMTRNNG::loadModel(NMTRNNG::Grad& grad, 
			const std::string& loadModelName, 
			const std::string& loadGradName) {
  this->load(loadModelName.c_str());
}

void NMTRNNG::saveResult(const Real value, 
			 const std::string& name) {
  /* For Model Analysis */
  std::ofstream valueFile;
  std::ostringstream ossValue;
  ossValue << this->saveDirName << "Model_NMTRNNG" << name;

  valueFile.open(ossValue.str(), std::ios::app); // open a file with 'a' mode

  valueFile << value << std::endl;
}

void NMTRNNG::demo(const std::string& srcTrain, 
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
		   const std::string& saveDirName) {
  Vocabulary sourceVoc(srcTrain, srcVocaThreshold, true);
  static Vocabulary targetVoc(tgtTrain, tgtVocaThreshold, true); 
  Vocabulary actionVoc(actTrain, true);

  std::vector<NMTRNNG::Data*> trainData, devData;

  NMTRNNG nmtRNNG(sourceVoc, 
		  targetVoc, 
		  actionVoc,
		  trainData, 
		  devData,
		  inputDim, 
		  inputActDim, 
		  hiddenEncDim, 
		  hiddenDim, 
		  hiddenActDim, 
		  scale,
		  useBlackOut, 
		  blackOutSampleNum, 
		  blackOutAlpha,
		  NMTRNNG::SGD, 
		  clipThreshold,
		  beamSize, 
		  maxLen, 
		  miniBatchSize, 
		  threadNum,
		  learningRate, 
		  false,
		  0, 
		  saveDirName);

  nmtRNNG.loadCorpus(srcTrain, tgtTrain, actTrain, trainData);
  nmtRNNG.loadCorpus(srcDev, tgtDev, actDev, devData); 

  std::cout << "# of Training Data:\t" << trainData.size() << std::endl;
  std::cout << "# of Development Data:\t" << devData.size() << std::endl;
  std::cout << "Source voc size: " << sourceVoc.tokenIndex.size() << std::endl;
  std::cout << "Target voc size: " << targetVoc.tokenIndex.size() << std::endl;
  std::cout << "Action voc size: " << actionVoc.tokenIndex.size() << std::endl;

  NMTRNNG::Grad grad(nmtRNNG);
  // Test
  auto test = trainData[0];
  for (int i = 0; i < 100; ++i) {
    std::cout << "\nEpoch " << i+1 
	      << " (lr = " << nmtRNNG.learningRate << ")" << std::endl;

    bool status = nmtRNNG.trainOpenMP(grad); 

    if (!status){
      nmtRNNG.load(nmtRNNG.prevModelFileName);
      nmtRNNG.learningRate *= 0.5;
      --i;
      continue;
    }

    // Save a model
    nmtRNNG.saveModel(grad, i);

    std::vector<NMTRNNG::ThreadArg> args;
    std::vector<std::vector<int> > translation(2);
    args.push_back(NMTRNNG::ThreadArg(nmtRNNG));
    args.push_back(NMTRNNG::ThreadArg(nmtRNNG));
    args[0].initTrans(nmtRNNG, 1);
    args[1].initTrans(nmtRNNG, 5);

    std::cout << "** Greedy Search" << std::endl;
    nmtRNNG.translate(test, args[0], translation[0], true);
    std::cout << "** Beam Search" << std::endl;
    nmtRNNG.translate(test, args[1], translation[1], true);
  }
}

void NMTRNNG::demo(const std::string& srcTrain, 
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
		   const int startIter) {
  Vocabulary sourceVoc(srcTrain, srcVocaThreshold, true);
  static Vocabulary targetVoc(tgtTrain, tgtVocaThreshold, true); 
  Vocabulary actionVoc(actTrain, true);

  std::vector<NMTRNNG::Data*> trainData, devData;

  NMTRNNG nmtRNNG(sourceVoc, 
		  targetVoc, 
		  actionVoc, 
		  trainData, 
		  devData, 
		  inputDim, 
		  inputActDim, 
		  hiddenDim, 
		  hiddenEncDim, 
		  hiddenActDim, 
		  scale,
		  useBlackOut, 
		  blackOutSampleNum, 
		  blackOutAlpha,
		  NMTRNNG::SGD, 
		  clipThreshold,
		  beamSize, 
		  maxLen, 
		  miniBatchSize, 
		  threadNum,
		  learningRate, 
		  false,
		  startIter, 
		  saveDirName);
  
  nmtRNNG.loadCorpus(srcTrain, tgtTrain, actTrain, trainData);
  nmtRNNG.loadCorpus(srcDev, tgtDev, actDev, devData); 

  std::vector<NMTRNNG::ThreadArg> args; // Evaluation of Dev.
  std::vector<std::vector<int> > translation(nmtRNNG.devData.size());
  for (int i = 0; i < threadNum; ++i){
    args.push_back(NMTRNNG::ThreadArg(nmtRNNG));
    args.back().initTrans(nmtRNNG, 1); // Sentences consists of less than 50 words
  }

  std::cout << "# of Training Data:\t" << trainData.size() << std::endl;
  std::cout << "# of Development Data:\t" << devData.size() << std::endl;
  std::cout << "Source voc size: " << sourceVoc.tokenIndex.size() << std::endl;
  std::cout << "Target voc size: " << targetVoc.tokenIndex.size() << std::endl;
  std::cout << "Action voc size: " << actionVoc.tokenIndex.size() << std::endl;

  NMTRNNG::Grad grad(nmtRNNG);
  // Model Loaded...
  nmtRNNG.loadModel(grad, loadModelName, loadGradName);
  nmtRNNG.prevModelFileName = loadModelName;
  nmtRNNG.prevGradFileName = loadGradName;

  // Test
  auto test = trainData[0];

  Real lossDev = 0.;
  Real lossActDev = 0.;
  Real tgtNum = 0.;
  Real actNum = 0.;

#pragma omp parallel for num_threads(nmtRNNG.threadNum) schedule(dynamic) // ThreadNum
  for (int i = 0; i < (int)devData.size(); ++i) {
    Real loss;
    Real lossAct;
    int id = omp_get_thread_num();
    std::tie(loss, lossAct) = nmtRNNG.calcLoss(devData[i], args[id], false);
#pragma omp critical
    {
      lossDev += loss;
      lossActDev += lossAct;
      tgtNum += devData[i]->tgt.size();
      actNum += devData[i]->action.size();
    }
  }

  Real currentDevPerp = exp(lossDev/tgtNum);
  std::cout << "[Language] Development Perplexity and Loss (/sentence):  " 
	    << currentDevPerp << ", "
	    << lossDev/devData.size() << "; "
	    << devData.size() << std::endl;
  std::cout << "[Action] Development Perplexity and Loss (/sentence):  " 
	    << exp(lossActDev/actNum) << ", "
	    << lossActDev/devData.size() << "; "
	    << devData.size() << std::endl;
  nmtRNNG.prevPerp = currentDevPerp;

  for (int i = 0; i < startIter; ++i) {
    nmtRNNG.rnd.shuffle(nmtRNNG.trainData);
  }
  for (int i = startIter; i < 100; ++i) {
    std::cout << "\nEpoch " << i+1 
	      << " (lr = " << nmtRNNG.learningRate 
	      << ")" << std::endl;
    
    bool status = nmtRNNG.trainOpenMP(grad); 
    if (!status){
      nmtRNNG.loadModel(grad, nmtRNNG.prevModelFileName, nmtRNNG.prevGradFileName);
      nmtRNNG.learningRate *= 0.5;
      --i;
      continue;
    }

    // Save a model
    nmtRNNG.saveModel(grad, i);

    std::vector<NMTRNNG::ThreadArg> argsTmp;
    std::vector<std::vector<int> > translation(2);
    argsTmp.push_back(NMTRNNG::ThreadArg(nmtRNNG));
    argsTmp.push_back(NMTRNNG::ThreadArg(nmtRNNG));
    argsTmp[0].initTrans(nmtRNNG, 1);
    argsTmp[1].initTrans(nmtRNNG, 5);

    std::cout << "** Greedy Search" << std::endl;
    nmtRNNG.translate(test, argsTmp[0], translation[0], true);
    std::cout << "** Beam Search" << std::endl;
    nmtRNNG.translate(test, args[1], translation[1], true);
  }
}

void NMTRNNG::evaluate(const std::string& srcTrain, 
		       const std::string& tgtTrain, 
		       const std::string& actTrain,
		       const std::string& srcTest, 
		       const std::string& tgtTest, 
		       const std::string& actTest,
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
		       const int maxLen,
		       const int miniBatchSize, 
		       const int threadNum,
		       const Real learningRate,
		       const int srcVocaThreshold, 
		       const int tgtVocaThreshold,
		       const bool isTest,
		       const std::string& saveDirName, 
		       const std::string& loadModelName, 
		       const std::string& loadGradName, 
		       const int startIter) {
  static Vocabulary sourceVoc(srcTrain, srcVocaThreshold, true);
  static Vocabulary targetVoc(tgtTrain, tgtVocaThreshold, true); 
  static Vocabulary actionVoc(actTrain, true); 
  static std::vector<NMTRNNG::Data*> trainData, testData;

  static NMTRNNG nmtRNNG(sourceVoc, 
			 targetVoc, 
			 actionVoc,
			 trainData, 
			 testData, 
			 inputDim, 
			 inputActDim, 
			 hiddenEncDim, 
			 hiddenDim, 
			 hiddenActDim, 
			 scale,
			 useBlackOut, 
			 blackOutSampleNum, 
			 blackOutAlpha, 
			 NMTRNNG::SGD, 
			 3.0,
			 beamSize, 
			 maxLen, 
			 miniBatchSize, 
			 threadNum,
			 learningRate, 
			 isTest,
			 startIter, 
			 saveDirName);
  
  if (testData.empty()) {
    nmtRNNG.loadCorpus(srcTest, tgtTest, actTest, testData); 
    std::cout << "# of Training Data:\t" << trainData.size() << std::endl;
    std::cout << "# of Evaluation Data:\t" << testData.size() << std::endl;
    std::cout << "Source voc size: " << sourceVoc.tokenIndex.size() << std::endl;
    std::cout << "Target voc size: " << targetVoc.tokenIndex.size() << std::endl;
    std::cout << "Action voc size: " << actionVoc.tokenIndex.size() << std::endl;
  } else {}
  std::vector<NMTRNNG::ThreadArg> args; // Evaluation of Test
  std::vector<std::vector<int> > translation(testData.size());
  for (int i = 0; i < threadNum; ++i) {
    args.push_back(NMTRNNG::ThreadArg(nmtRNNG));
    args.back().initTrans(nmtRNNG, nmtRNNG.beamSize);
  }

  NMTRNNG::Grad grad(nmtRNNG);
  // Model Loaded...
  nmtRNNG.loadModel(grad, loadModelName, loadGradName);

  Real lossTest = 0.;
  Real lossActTest = 0.;
  Real tgtNum = 0.;
  Real actNum = 0.;

#pragma omp parallel for num_threads(nmtRNNG.threadNum) schedule(dynamic) shared(args) // ThreadNum
  for (int i = 0; i < (int)testData.size(); ++i) {
    Real loss;
    Real lossAct;
    int id = omp_get_thread_num();
    std::tie(loss, lossAct) = nmtRNNG.calcLoss(testData[i], args[id], false);
#pragma omp critical
    {
      lossTest += loss;
      lossActTest += lossAct;
      tgtNum += testData[i]->tgt.size(); // include `*EOS*`
      actNum += testData[i]->action.size();
    }
  }
  std::cout << "[Language] Perplexity and Loss (/sentence):  " 
	    << exp(lossTest/tgtNum) << ", "
	    << lossTest/testData.size() << "; "
	    << testData.size() << std::endl;

  std::cout << "[Action] Perplexity and Loss (/sentence):  " 
	    << exp(lossActTest/actNum) << ", "
	    << lossActTest/testData.size() << "; "
	    << testData.size() << std::endl;
 
#pragma omp parallel for num_threads(nmtRNNG.threadNum) schedule(dynamic) // ThreadNum
  for (int i = 0; i < (int)testData.size(); ++i) {
    auto evalData = testData[i];
    int id = omp_get_thread_num();
    nmtRNNG.translate(evalData, args[id], translation[i], false);
    // nmtRNNG.translateWithAction(evalData, args[id], translation[i], 1, false);
  }

  std::ofstream outputFile;
  std::ostringstream oss;
  std::string parsedMode;
  oss << nmtRNNG.saveDirName << "Model_NMTRNNG"
      << ".BlackOut_" << (nmtRNNG.useBlackOut?"true":"false")
      << ".beamSize_" << nmtRNNG.beamSize 
      << ".lrSGD_" << nmtRNNG.learningRate 
      << ".startIter_" << startIter
      << ".Output" << (nmtRNNG.isTest?"Test":"Dev")
      << ".translate";
  outputFile.open(oss.str(), std::ios::out);

  for (int i = 0; i < (int)testData.size(); ++i) {
    auto evalData = testData[i];
    for (auto it = evalData->trans.begin(); it != evalData->trans.end(); ++it) {
      outputFile << nmtRNNG.targetVoc.tokenList[*it]->str << " ";
    }
    outputFile << std::endl;
    // trans
    testData[i]->trans.clear();
  }
}

void NMTRNNG::save(const std::string& fileName) {
  std::ofstream ofs(fileName.c_str(), std::ios::out|std::ios::binary);
  assert(ofs);

  this->enc.save(ofs);
  this->encRev.save(ofs);
  this->dec.save(ofs);
  this->act.save(ofs);
  this->outBuf.save(ofs);

  this->decInitAffine.save(ofs);
  this->actInitAffine.save(ofs);
  this->outBufInitAffine.save(ofs);
  this->stildeAffine.save(ofs);
  this->utAffine.save(ofs);
  this->embedVecAffine.save(ofs);

  this->softmax.save(ofs);
  this->blackOut.save(ofs);
  this->softmaxAct.save(ofs);

  Utils::save(ofs, sourceEmbed);
  Utils::save(ofs, targetEmbed);
  Utils::save(ofs, actionEmbed);

  Utils::save(ofs, Wgeneral);
}

void NMTRNNG::load(const std::string& fileName) {
  std::ifstream ifs(fileName.c_str(), std::ios::in|std::ios::binary);
  assert(ifs);

  this->enc.load(ifs);
  this->encRev.load(ifs);
  this->dec.load(ifs);
  this->act.load(ifs);
  this->outBuf.load(ifs);

  this->decInitAffine.load(ifs);
  this->actInitAffine.load(ifs);
  this->outBufInitAffine.load(ifs);
  this->stildeAffine.load(ifs);
  this->utAffine.load(ifs);
  this->embedVecAffine.load(ifs);

  this->softmax.load(ifs);
  this->blackOut.load(ifs);
  this->softmaxAct.load(ifs);

  Utils::load(ifs, sourceEmbed);
  Utils::load(ifs, targetEmbed);
  Utils::load(ifs, actionEmbed);

  Utils::load(ifs, Wgeneral);
}

/* NMTRNNG::DecCandidate */
void NMTRNNG::DecCandidate::init(NMTRNNG& nmtRNNG) {
  this->k = 0;
  this->score = 0.0;
  this->scoreAct = 0.0;
  this->generatedTarget.clear();
  this->generatedAction.clear();
  this->stop = false;
  this->phraseNum = 150;
  this->tgtLen = 150;


  while(!this->headStack.empty()) {
    this->headStack.pop();
  }
  while (!this->embedStack.empty()) {
    this->embedStack.pop();
  }

  if (this->decState.empty()) {
    for (int i = 0; i < 150; ++i) {
      // Vectors or Matrices
      this->embedVec.push_back(VecD(nmtRNNG.inputDim));
    }

     for (int i = 0; i < nmtRNNG.maxLen; ++i) {
      LSTM::State *lstmDecState(NULL);
      lstmDecState = new LSTM::State;
      this->decState.push_back(lstmDecState);
      LSTM::State *lstmActState(NULL);
      lstmActState = new LSTM::State;
      this->actState.push_back(lstmActState);
      LSTM::State *lstmOutBufState(NULL);
      lstmOutBufState = new LSTM::State;
      this->outBufState.push_back(lstmOutBufState);
    }

    int utSize = nmtRNNG.hiddenDim*2 + nmtRNNG.hiddenActDim;
    int embedVecSize = nmtRNNG.inputDim*2 + nmtRNNG.inputActDim;
    this->utEnd = VecD(utSize);
    for (int i = 0; i < nmtRNNG.maxLen*2; ++i) {
      // Affine Vecors
      this->embedVecEnd.push_back(VecD(embedVecSize));
    }
  }
}

/* NMTRNNG::ThreadFunc */
NMTRNNG::ThreadArg::ThreadArg(NMTRNNG& nmtRNNG) {
  // LSTM
  int stildeSize = nmtRNNG.hiddenDim + nmtRNNG.hiddenEncDim*2;
  for (int i = 0; i < 150; ++i) {
    LSTM::State *lstmState(NULL);
    lstmState = new LSTM::State;
    this->encState.push_back(lstmState);
    LSTM::State *lstmStateRev(NULL);
    lstmStateRev = new LSTM::State;
    this->encRevState.push_back(lstmStateRev);
    this->biEncState.push_back(VecD(nmtRNNG.hiddenEncDim*2));

    // Vectors or Matrices
    this->s_tilde.push_back(VecD(nmtRNNG.hiddenDim));
    this->embedVec.push_back(VecD(nmtRNNG.inputDim));
    this->contextSeqList.push_back(VecD(nmtRNNG.hiddenEncDim*2));
    this->del_stilde.push_back(VecD(nmtRNNG.hiddenDim));
    // Affine
    this->stildeEnd.push_back(VecD(stildeSize));
  }

  for (int i = 0; i < nmtRNNG.maxLen; ++i) {
    LSTM::State *lstmDecState(NULL);
    lstmDecState = new LSTM::State;
    this->decState.push_back(lstmDecState);
  }
  for (int i = 0; i < nmtRNNG.maxLen*2; ++i) {
    LSTM::State *lstmActState(NULL);
    lstmActState = new LSTM::State;
    this->actState.push_back(lstmActState);
    LSTM::State *lstmOutBufState(NULL);
    lstmOutBufState = new LSTM::State;
    this->outBufState.push_back(lstmOutBufState);
  }

  if (nmtRNNG.useBlackOut){
    for (int i= 0; i < nmtRNNG.maxLen; ++i) {
      this->blackOutState.push_back(BlackOut::State(nmtRNNG.blackOut));
      this->targetDistVec.push_back(VecD());
    }
  }

  int utSize = nmtRNNG.hiddenDim*2 + nmtRNNG.hiddenActDim;
  int embedVecSize = nmtRNNG.inputDim*2 + nmtRNNG.inputActDim;
  for (int i = 0; i < nmtRNNG.maxLen*2; ++i) {
    // Vectors or Matrices
    this->ut.push_back(VecD(nmtRNNG.hiddenDim));
    this->del_ut.push_back(VecD(utSize));
    // Affine Vecors
    this->utEnd.push_back(VecD(utSize));
    this->embedVecEnd.push_back(VecD(embedVecSize));
  }

  // Vectors or Matrices
  this->encStateEnd = VecD(nmtRNNG.hiddenEncDim*2);
}

void NMTRNNG::ThreadArg::initTrans(NMTRNNG& nmtRNNG, 
				   const int beamSize) {
  for (int i = 0; i < beamSize; ++i) {
    this->candidate.push_back(NMTRNNG::DecCandidate());
    this->candidate.back().init(nmtRNNG);
  }
}

void NMTRNNG::ThreadArg::clear() {
  this->del_embedVec.clear();
  while(!headStack.empty()) {
    headStack.pop();
  }
  headList.clear();
  while (!embedStack.empty()) {
    embedStack.pop();
  }
  embedList.clear();
}

void NMTRNNG::ThreadArg::initLoss() {
  this->loss = 0.0;
  this->lossAct = 0.0;
}

void NMTRNNG::ThreadArg::init(NMTRNNG& nmtRNNG, 
			      const NMTRNNG::Data* data, 
			      const bool train) {
  this->srcLen = data->src.size();
  this->tgtLen = data->tgt.size();
  this->actLen = data->action.size();

  if (train) {
    this->alphaSeq = MatD::Zero(this->srcLen, this->tgtLen);
    this->encStateEnd = nmtRNNG.zeros2;
    this->del_encStateEnd = nmtRNNG.zeros2;
    this->del_alphaSeq = VecD(this->srcLen);
    this->del_alphaSeqTmp = nmtRNNG.zeros;
    this->del_WgeneralTmp = nmtRNNG.zeros2;
    this->alphaSeqVec = VecD(this->srcLen);

  } else {
    this->alphaSeqVec = VecD(this->srcLen);
  }
}

/* NMTRNNG::Grad */
NMTRNNG::Grad::Grad(NMTRNNG& nmtRNNG):
  gradHist(0)
{
  this->lstmSrcGrad = LSTM::Grad(nmtRNNG.enc);
  this->lstmSrcRevGrad = LSTM::Grad(nmtRNNG.encRev);
  this->lstmTgtGrad = LSTM::Grad(nmtRNNG.dec);
  this->lstmActGrad = LSTM::Grad(nmtRNNG.act);
  this->lstmOutBufGrad = LSTM::Grad(nmtRNNG.outBuf);

  this->decInitAffineGrad = Affine::Grad(nmtRNNG.decInitAffine);
  this->actInitAffineGrad = Affine::Grad(nmtRNNG.actInitAffine);
  this->outBufInitAffineGrad = Affine::Grad(nmtRNNG.outBufInitAffine);
  this->stildeAffineGrad = Affine::Grad(nmtRNNG.stildeAffine);
  this->utAffineGrad = Affine::Grad(nmtRNNG.utAffine);
  this->embedVecAffineGrad = Affine::Grad(nmtRNNG.embedVecAffine);

  if (!nmtRNNG.useBlackOut) {
    this->softmaxGrad = SoftMax::Grad(nmtRNNG.softmax);
  } else {
    this->blackOutGrad = BlackOut::Grad(nmtRNNG.blackOut, false);
  }
  this->softmaxActGrad = SoftMax::Grad(nmtRNNG.softmaxAct);

  this->Wgeneral = MatD::Zero(nmtRNNG.Wgeneral.rows(), nmtRNNG.Wgeneral.cols());

  this->init();
}

void NMTRNNG::Grad::init() {
    this->sourceEmbed.clear();
    this->targetEmbed.clear();
    this->actionEmbed.clear();

    this->lstmSrcGrad.init();
    this->lstmSrcRevGrad.init();
    this->lstmTgtGrad.init();
    this->lstmActGrad.init();
    this->lstmOutBufGrad.init();

    this->decInitAffineGrad.init();
    this->actInitAffineGrad.init();
    this->outBufInitAffineGrad.init();
    this->stildeAffineGrad.init();
    this->utAffineGrad.init();
    this->embedVecAffineGrad.init();

    this->softmaxGrad.init();
    this->softmaxActGrad.init();
    this->blackOutGrad.init();

    this->Wgeneral.setZero();
}

Real NMTRNNG::Grad::norm() {
  Real res = 0.0; 

  for (auto it = this->sourceEmbed.begin(); it != this->sourceEmbed.end(); ++it){
    res += it->second.squaredNorm();
  }
  for (auto it = this->targetEmbed.begin(); it != this->targetEmbed.end(); ++it){
    res += it->second.squaredNorm();
  }
  for (auto it = this->actionEmbed.begin(); it != this->actionEmbed.end(); ++it){
    res += it->second.squaredNorm();
  }

  res += this->lstmSrcGrad.norm();
  res += this->lstmSrcRevGrad.norm();
  res += this->lstmTgtGrad.norm();
  res += this->lstmActGrad.norm();
  res += this->lstmOutBufGrad.norm();

  res += this->decInitAffineGrad.norm();
  res += this->actInitAffineGrad.norm();
  res += this->outBufInitAffineGrad.norm();
  res += this->stildeAffineGrad.norm();
  res += this->utAffineGrad.norm();
  res += this->embedVecAffineGrad.norm();

  res += this->softmaxGrad.norm();
  res += this->softmaxActGrad.norm();
  res += this->blackOutGrad.norm();

  res += this->Wgeneral.squaredNorm();

  return res;
}

void NMTRNNG::Grad::operator += (const NMTRNNG::Grad& grad) {
  for (auto it = grad.sourceEmbed.begin(); it != grad.sourceEmbed.end(); ++it){
    if (this->sourceEmbed.count(it->first)){
      this->sourceEmbed.at(it->first) += it->second;
    } else {
      this->sourceEmbed[it->first] = it->second;
    }
  }

  for (auto it = grad.targetEmbed.begin(); it != grad.targetEmbed.end(); ++it){
    if (this->targetEmbed.count(it->first)){
      this->targetEmbed.at(it->first) += it->second;
    } else {
      this->targetEmbed[it->first] = it->second;
    }
  }

  for (auto it = grad.actionEmbed.begin(); it != grad.actionEmbed.end(); ++it){
    if (this->actionEmbed.count(it->first)){
      this->actionEmbed.at(it->first) += it->second;
    } else {
      this->actionEmbed[it->first] = it->second;
    }
  }

  this->lstmSrcGrad += grad.lstmSrcGrad;
  this->lstmSrcRevGrad += grad.lstmSrcRevGrad;
  this->lstmTgtGrad += grad.lstmTgtGrad;
  this->lstmActGrad += grad.lstmActGrad;
  this->lstmOutBufGrad += grad.lstmOutBufGrad;

  this->decInitAffineGrad += grad.decInitAffineGrad;
  this->actInitAffineGrad += grad.actInitAffineGrad;
  this->outBufInitAffineGrad += grad.outBufInitAffineGrad;
  this->stildeAffineGrad += grad.stildeAffineGrad;
  this->utAffineGrad += grad.utAffineGrad;
  this->embedVecAffineGrad += grad.embedVecAffineGrad;

  this->softmaxGrad += grad.softmaxGrad;  
  this->softmaxActGrad += grad.softmaxActGrad;  
  this->blackOutGrad += grad.blackOutGrad;  

  this->Wgeneral += grad.Wgeneral;
}

void NMTRNNG::Grad::sgd(NMTRNNG& nmtRNNG, const Real learningRate) {
  for (auto it = this->sourceEmbed.begin(); it != this->sourceEmbed.end(); ++it) {
    nmtRNNG.sourceEmbed.col(it->first) -= learningRate * it->second;
  }
  for (auto it = this->targetEmbed.begin(); it != this->targetEmbed.end(); ++it) {
    nmtRNNG.targetEmbed.col(it->first) -= learningRate * it->second;
  }
  for (auto it = this->actionEmbed.begin(); it != this->actionEmbed.end(); ++it) {
    nmtRNNG.actionEmbed.col(it->first) -= learningRate * it->second;
  }

  this->lstmSrcGrad.sgd(learningRate, nmtRNNG.enc);
  this->lstmSrcRevGrad.sgd(learningRate, nmtRNNG.encRev);
  this->lstmTgtGrad.sgd(learningRate, nmtRNNG.dec);
  this->lstmActGrad.sgd(learningRate, nmtRNNG.act);
  this->lstmOutBufGrad.sgd(learningRate, nmtRNNG.outBuf);

  this->decInitAffineGrad.sgd(learningRate, nmtRNNG.decInitAffine);
  this->actInitAffineGrad.sgd(learningRate, nmtRNNG.actInitAffine);
  this->outBufInitAffineGrad.sgd(learningRate, nmtRNNG.outBufInitAffine);
  this->stildeAffineGrad.sgd(learningRate, nmtRNNG.stildeAffine);
  this->utAffineGrad.sgd(learningRate, nmtRNNG.utAffine);
  this->embedVecAffineGrad.sgd(learningRate, nmtRNNG.embedVecAffine);

  if (!nmtRNNG.useBlackOut) {
    this->softmaxGrad.sgd(learningRate, nmtRNNG.softmax);
  } else {
    this->blackOutGrad.sgd(learningRate, nmtRNNG.blackOut);
  }
  this->softmaxActGrad.sgd(learningRate, nmtRNNG.softmaxAct);

  Optimizer::sgd(this->Wgeneral, learningRate, nmtRNNG.Wgeneral);
}

void NMTRNNG::Grad::save(NMTRNNG& nmtRNNG, 
			 const std::string& filename) {
  std::ofstream ofs(filename.c_str(), std::ios::out|std::ios::binary);
  assert(ofs);

  Utils::save(ofs, this->gradHist->sourceEmbedMatGrad);
  Utils::save(ofs, this->gradHist->targetEmbedMatGrad);
  Utils::save(ofs, this->gradHist->actionEmbedMatGrad);

  // LSTM
  this->lstmSrcGrad.saveHist(ofs);
  this->lstmSrcRevGrad.saveHist(ofs);
  this->lstmTgtGrad.saveHist(ofs);
  this->lstmActGrad.saveHist(ofs);
  this->lstmOutBufGrad.saveHist(ofs);

  // Affine
  this->decInitAffineGrad.saveHist(ofs);
  this->actInitAffineGrad.saveHist(ofs);
  this->outBufInitAffineGrad.saveHist(ofs);
  this->utAffineGrad.saveHist(ofs);
  this->stildeAffineGrad.saveHist(ofs);
  this->embedVecAffineGrad.saveHist(ofs);

  this->softmaxActGrad.saveHist(ofs);
  if (nmtRNNG.useBlackOut) {
    this->blackOutGrad.saveHist(ofs);
  } else {
    this->softmaxGrad.saveHist(ofs);
  }

  Utils::save(ofs, this->gradHist->WgeneralMatGrad);
}

void NMTRNNG::Grad::load(NMTRNNG& nmtRNNG, 
			 const std::string& filename) {
  std::ifstream ifs(filename.c_str(), std::ios::in|std::ios::binary);
  assert(ifs);

  Utils::load(ifs, this->gradHist->sourceEmbedMatGrad);
  Utils::load(ifs, this->gradHist->targetEmbedMatGrad);
  Utils::load(ifs, this->gradHist->actionEmbedMatGrad);

  // LSTM
  this->lstmSrcGrad.loadHist(ifs);
  this->lstmSrcRevGrad.loadHist(ifs);
  this->lstmTgtGrad.loadHist(ifs);
  this->lstmActGrad.loadHist(ifs);
  this->lstmOutBufGrad.loadHist(ifs);

  // Affine
  this->decInitAffineGrad.loadHist(ifs);
  this->actInitAffineGrad.loadHist(ifs);
  this->outBufInitAffineGrad.loadHist(ifs);
  this->utAffineGrad.loadHist(ifs);
  this->stildeAffineGrad.loadHist(ifs);
  this->embedVecAffineGrad.loadHist(ifs);

  this->softmaxActGrad.loadHist(ifs);
  if (nmtRNNG.useBlackOut) {
    this->blackOutGrad.loadHist(ifs);
  } else {
    this->softmaxGrad.loadHist(ifs);
  }
  
  Utils::load(ifs, this->gradHist->WgeneralMatGrad);
}
